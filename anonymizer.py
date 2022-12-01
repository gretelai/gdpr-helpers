import json
import os
import pickle
from typing import TYPE_CHECKING, Optional

import pandas as pd
import yaml
from pathlib import Path
from smart_open import open

from gretel_client import configure_session, poll, submit_docker_local
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config


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
        self._cache_init_report = Path(self.tmp_dir / "init_report.pkl")
        self._cache_run_report = Path(self.tmp_dir / "run_report.pkl")
        self._cache_syn_report = Path(self.tmp_dir / "syn_report.pkl")
        self.dataset_path: Optional[Path] = None
        self.deid_df = None
        self.synthetic_df = None
        self.init_report = {}
        self.run_report = {}
        self.syn_report = {}

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

    def _transform_hybrid(self, config:dict):
        """Gretel hybrid cloud API."""
        df = pd.read_csv(self.training_path)
        df.head(self.preview_recs).to_csv(self.preview_path, index=False)
        transform_train = self.project.create_model_obj(config, str(self.preview_path))
        run = submit_docker_local(
            transform_train,
            output_dir=str(self.tmp_dir),
        )
        self.init_report = json.loads(open(self.tmp_dir / "report_json.json.gz").read())

        # Use model to transform records
        transform_go = transform_train.create_record_handler_obj(
            data_source=self.training_path
        )
        run = submit_docker_local(
            transform_go, model_path=self.tmp_dir / "model.tar.gz", output_dir=self.tmp_dir
        )
        self.run_report = json.loads(open(self.tmp_dir / "report_json.json.gz").read())
        self.deid_df = pd.read_csv(self.tmp_dir / "data.gz")
        self.deid_df.to_csv(self.deid_path, index=False)

    def _transform_cloud(self, config:dict):
        """Gretel SaaS API."""
        df = pd.read_csv(self.training_path)
        model = self.project.create_model_obj(
            config, 
            data_source=df.head(self.preview_recs)
        )
        model.submit_cloud()
        poll(model)
        with open(model.get_artifact_link("report_json")) as fh:
            self.init_report = json.loads(fh.read())

        # Use model to transform records
        rh = model.create_record_handler_obj(data_source=df)
        rh.submit_cloud()
        poll(rh)
        with open(rh.get_artifact_link("run_report_json")) as fh:
            self.run_report = json.loads(fh.read())
        self.deid_df = pd.read_csv(rh.get_artifact_link("data"), compression="gzip")
        self.deid_df.to_csv(self.deid_path, index=False)

    def _print_ner_report(self):
        """Save a markdown-format report of the NER findings from a dataset."""
        report = self.init_report
        df = pd.DataFrame(report["metadata"]["fields"])[
            ["name", "count", "approx_distinct_count", "missing_count", "labels"]
        ]
        df["labels"] = df["labels"].astype(str).replace("[\[\]']", "", regex=True)
        df.rename(
            columns={"labels": "entities_detected", "name": "column_name"}, inplace=True
        )
        report_content = (
            "\n\nNamed Entity Recognition (NER) finished.\n"
            f"Processing time: {report['training_time_seconds']} seconds\n"
            f"Record count: {report['record_count']} records\n"
            f"Column count: {report['field_count']} columns\n\n"
            f"Dataset overview\n"
            f"{df.to_markdown(index = False)}\n\n"
        )
        with open(self.deid_report_path, "w") as fh:
            fh.write(report_content)
        print(report_content)

    def _print_transform_report(self):
        """Save a markdown-format report of data transformations on a dataset."""
        report = self.run_report
        report_content = (
            "Transforms finished.\n"
            f"Processing time: {report['summary'][0]['value']} seconds\n"
            f"Record count: {report['summary'][1]['value']}\n\n"
            f"Columns transformed\n"
            f"{pd.DataFrame(report['summary'][2]['value']).to_markdown(index = False)}\n\n"
        )
        with open(self.deid_report_path, "a") as fh:
            fh.write(report_content)
        print(report_content)

    
    def _print_synthesis_report(self):
        """Save a markdown-format report of data synthesis on a dataset."""
        report = self.syn_report
        print(json.dumps(report, indent=2))
        exit(0)
        report_content = (
            "Synthesis finished.\n"
            f"Processing time: {report['summary'][0]['value']} seconds\n"
            f"Record count: {report['summary'][1]['value']}\n\n"
            f"Columns transformed\n"
            f"{pd.DataFrame(report['summary'][2]['value']).to_markdown(index = False)}\n\n"
        )
        #with open(self.deid_report_path, "a") as fh:
        #    fh.write(report_content)
        print(report_content)

    def transform(self):
        """Deidentify a dataset using Gretel's Transform APIs."""
        config = yaml.safe_load(self.tx_config)

        if self._cache_init_report.exists() and self._cache_run_report.exists():
            self.init_report = pickle.load(open(self._cache_init_report, "rb"))
            self.run_report = pickle.load(open(self._cache_run_report, "rb"))
            self.deid_df = pd.read_csv(self.deid_path)
        else:
            # Initialize transform model
            if self.run_mode == "cloud":
                self._transform_cloud(config=config)
            elif self.run_mode == "hybrid":
                self._transform_hybrid(config=config)

            pickle.dump(self.init_report, open(self._cache_init_report, "wb"))
            pickle.dump(self.run_report, open(self._cache_run_report, "wb"))
            self.deid_df.to_csv(self.deid_path, index=False)

        self._print_ner_report()
        self._print_transform_report()

    def synthesize(self):
        """Train a synthetic data model on a dataset and use it to create an artificial
        version of a dataset with increased privacy guarantees.
        """
        config = yaml.safe_load(self.sx_config)
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

        #self._print_synthesis_report()

    def _synthesize_cloud(self, config:dict):
        """Gretel SaaS APIs.
        """
        model = self.project.create_model_obj(
            model_config=config, 
            data_source=str(self.deid_path)
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

    def _synthesize_hybrid(self, config:dict):
        """Gretel Hybrid Cloud APIs"""
        model = self.project.create_model_obj(model_config=config)
        run = submit_docker_local(model, output_dir=str(self.tmp_dir))

        self.synthetic_df = pd.read_csv(self.tmp_dir / "data_preview.gz", compression="gzip")
        self.synthetic_df.to_csv(self.anonymized_path, index=False)
        self.syn_report = json.loads(open(self.tmp_dir / "report_json.json.gz").read())
        pickle.dump(self.syn_report, open(self._cache_syn_report, "wb"))
