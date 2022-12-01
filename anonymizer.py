import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import yaml
from smart_open import open

from gretel_client import configure_session, poll, submit_docker_local
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config

import reports

PREVIEW_RECS = 100


class Anonymizer:
    """Automated model training and synthetic data generation tool
    Args:
        project_name (str, optional): Gretel project name. Defaults to "gdpr-anonymized".
        overwrite (bool, optional): Overwrite previous progress. Defaults to True.
    """

    def __init__(
        self,
        project_name: str = "gdpr-anonymized",
        tx_config: str = "./config/transform_config.yaml",
        sx_config: str = "./config/synthetics_config.yaml",
        run_mode: str = "cloud",
        preview_recs: int = PREVIEW_RECS,
        output_dir: str = "artifacts",
        tmp_dir: str = "tmp",
    ):
        configure_session(api_key="prompt", cache="yes", validate=True)

        self.project_name = project_name
        self.sx_config = sx_config
        self.tx_config = tx_config
        self.run_mode = run_mode
        self.preview_recs = preview_recs
        self.output_dir = Path(output_dir)
        self.tmp_dir = Path(tmp_dir)

        self.project = create_or_get_unique_project(name=project_name)
        self.deid_path = Path(self.output_dir / "deidentified_data.csv")
        self.deid_report_path = Path(self.output_dir / "deidentification_report.md")
        self.anonymized_path = Path(self.output_dir / "synthetic_data.csv")
        self.training_path = Path(self.tmp_dir / "training_data.csv")
        self.preview_path = Path(self.tmp_dir / "preview.csv")
        self._cache_ner_report = Path(self.tmp_dir / "ner_report.pkl")
        self._cache_run_report = Path(self.tmp_dir / "run_report.pkl")
        self._cache_syn_report = Path(self.tmp_dir / "syn_report.pkl")
        self.dataset_path: Optional[Path] = None
        self.deid_df = None
        self.synthetic_df = None
        self.ner_report = {}
        self.run_report = {}
        self.syn_report = {}

        assert self.run_mode in [
            "cloud",
            "hybrid",
        ], "Error: run_mode param must be either 'cloud' or 'hybrid"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def anonymize(self, dataset_path: str):
        """Anonymize a dataset to GDPR standards
        using a pipeline of named entity recognition, data transformations,
        and synthetic data model training and generation"

        Args:
            dataset_path (str): Path or URL to CSV
        """
        self._preprocess_data(dataset_path)
        self.transform()
        self.synthesize()
        self._save_reports(self.deid_report_path)

        print("Anonymization complete")
        print(f" -- Synthetic data stored to: {self.anonymized_path}")
        print(f" -- Anonymization report stored to: {self.deid_report_path}")

    def _save_reports(self, output_path: Path):
        """Save anonymization reports to a local file in markdown format
        """
        report = (
            f"{reports.ner_report(self.ner_report)}"
            f"{reports.transform_report(self.run_report)}"
            f"{reports.synthesis_report(self.syn_report)}"
        )
        with open(self.deid_report_path, "w") as fh:
            fh.write(report)

    def _preprocess_data(self, ds: str) -> str:
        """Remove NaNs from input data before training model.

        Args:
            ds (str): Path to source dataset
        """
        df = pd.read_csv(ds)
        nan_columns = df.columns[df.isna().any()].tolist()
        print(f"Warning: Found NaN values in training data columns: {nan_columns}")
        df = df.fillna("")
        df.to_csv(self.training_path, index=False)

    def _transform_hybrid(self, config: dict):
        """Gretel hybrid cloud API."""
        df = pd.read_csv(self.training_path)
        df.head(self.preview_recs).to_csv(self.preview_path, index=False)
        transform_train = self.project.create_model_obj(config, str(self.preview_path))
        run = submit_docker_local(transform_train, output_dir=str(self.tmp_dir),)
        self.ner_report = json.loads(open(self.tmp_dir / "report_json.json.gz").read())

        # Use model to transform records
        transform_go = transform_train.create_record_handler_obj(
            data_source=str(self.training_path)
        )
        run = submit_docker_local(
            transform_go,
            model_path=str(self.tmp_dir / "model.tar.gz"),
            output_dir=str(self.tmp_dir),
        )
        self.run_report = json.loads(open(self.tmp_dir / "report_json.json.gz").read())
        self.deid_df = pd.read_csv(self.tmp_dir / "data.gz")
        self.deid_df.to_csv(self.deid_path, index=False)

    def _transform_cloud(self, config: dict):
        """Gretel SaaS API."""
        df = pd.read_csv(self.training_path)
        model = self.project.create_model_obj(
            config, data_source=df.head(self.preview_recs)
        )
        model.submit_cloud()
        poll(model)
        with open(model.get_artifact_link("report_json")) as fh:
            self.ner_report = json.loads(fh.read())

        # Use model to transform records
        rh = model.create_record_handler_obj(data_source=df)
        rh.submit_cloud()
        poll(rh)
        with open(rh.get_artifact_link("run_report_json")) as fh:
            self.run_report = json.loads(fh.read())
        self.deid_df = pd.read_csv(rh.get_artifact_link("data"), compression="gzip")
        self.deid_df.to_csv(self.deid_path, index=False)

    def transform(self):
        """Deidentify a dataset using Gretel's Transform APIs."""
        with open(self.tx_config, "r") as stream:
            config = yaml.safe_load(stream)

        if self._cache_ner_report.exists() and self._cache_run_report.exists():
            self.ner_report = pickle.load(open(self._cache_ner_report, "rb"))
            self.run_report = pickle.load(open(self._cache_run_report, "rb"))
            self.deid_df = pd.read_csv(self.deid_path)
        else:
            # Initialize transform model
            if self.run_mode == "cloud":
                self._transform_cloud(config=config)
            elif self.run_mode == "hybrid":
                self._transform_hybrid(config=config)

            pickle.dump(self.ner_report, open(self._cache_ner_report, "wb"))
            pickle.dump(self.run_report, open(self._cache_run_report, "wb"))
            self.deid_df.to_csv(self.deid_path, index=False)

        print(reports.ner_report(self.ner_report))
        print(reports.transform_report(self.run_report))

    def synthesize(self):
        """Train a synthetic data model on a dataset and use it to create an artificial
        version of a dataset with increased privacy guarantees.
        """
        with open(self.sx_config, "r") as stream:
            config = yaml.safe_load(stream)

        config["models"][0]["actgan"]["generate"] = {"num_records": len(self.deid_df)}
        config["models"][0]["actgan"]["data_source"] = str(self.training_path)

        if self._cache_syn_report.exists():
            self.syn_report = pickle.load(open(self._cache_syn_report, "rb"))
            self.synthetic_df = pd.read_csv(self.anonymized_path)
        else:
            if self.run_mode == "cloud":
                self._synthesize_cloud(config=config)
            elif self.run_mode == "hybrid":
                self._synthesize_hybrid(config=config)

        print(reports.synthesis_report(self.syn_report))

    def _synthesize_cloud(self, config: dict):
        """Gretel SaaS APIs.
        """
        model = self.project.create_model_obj(
            model_config=config, data_source=str(self.deid_path)
        )
        model.submit_cloud()
        poll(model)
        self.synthetic_df = pd.read_csv(
            model.get_artifact_link("data_preview"), compression="gzip"
        )
        self.synthetic_df.to_csv(self.anonymized_path, index=False)
        with open(model.get_artifact_link("report_json")) as fh:
            self.syn_report = json.loads(fh.read())
            pickle.dump(self.syn_report, open(self._cache_syn_report, "wb"))

    def _synthesize_hybrid(self, config: dict):
        """Gretel Hybrid Cloud APIs"""
        model = self.project.create_model_obj(model_config=config)
        run = submit_docker_local(model, output_dir=str(self.tmp_dir))
        self.synthetic_df = pd.read_csv(
            self.tmp_dir / "data_preview.gz", compression="gzip"
        )
        self.synthetic_df.to_csv(self.anonymized_path, index=False)
        self.syn_report = json.loads(open(self.tmp_dir / "report_json.json.gz").read())
        pickle.dump(self.syn_report, open(self._cache_syn_report, "wb"))
