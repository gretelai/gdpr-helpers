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
        run_mode: str = "cloud",
        output_dir: str = "artifacts",
        preview_recs: int = PREVIEW_RECS,
    ):
        configure_session(api_key="prompt", cache="yes", validate=True)

        self.project_name = project_name
        self.tx_config = tx_config
        self.run_mode = run_mode
        self.preview_recs = preview_recs
        self.output_dir = Path(output_dir)

        self.project = create_or_get_unique_project(name=project_name)
        self.training_path = Path(self.output_dir / "training_data.csv")
        self.anonymized_path = Path(self.output_dir / "synthetic_data.csv")
        self.preview_path = Path(self.output_dir / "tmp-preview.csv")
        self.deid_path = Path(self.output_dir / "deidentified_data.csv")
        self.deid_report_path = Path(self.output_dir / "deidentification_report.md")
        self._cache_init_report = Path(self.output_dir / "init_report.pkl")
        self._cache_run_report = Path(self.output_dir / "run_report.pkl")
        self._cache_syn_report = Path(self.output_dir / "syn_report.pkl")
        self.dataset_path: Optional[Path] = None
        self.deid_df = None
        self.synthetic_df = None
        self.init_report = {}
        self.run_report = {}
        self.syn_report = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def anonymize(self, dataset_path: str):
        """Anonymize a dataset to GDPR standards
        using a pipeline of named entity recognition, data transformations,
        and synthetic data model training and generation"

        Args:
            dataset_path (str): Path or URL to CSV
        """
        self._preprocess_data(dataset_path)
        self.transform()
        # self.synthesize()

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

    def _transform_local(self):
        """Deidentify a dataset using Gretel's hybrid cloud API."""
        df = pd.read_csv(self.training_path)
        df.head(self.preview_recs).to_csv(self.preview_path, index=False)
        config = yaml.safe_load(self.tx_config)
        transform_train = self.project.create_model_obj(config, self.preview_path)
        run = submit_docker_local(
            transform_train,
            output_dir="tmp/",
        )
        self.init_report = json.loads(open("tmp/report_json.json.gz").read())

        # Use model to transform records
        transform_go = transform_train.create_record_handler_obj(
            data_source=self.training_path
        )
        run = submit_docker_local(
            transform_go, model_path="tmp/model.tar.gz", output_dir="tmp/"
        )
        self.run_report = json.loads(open("tmp/report_json.json.gz").read())
        self.deid_df = pd.read_csv("tmp/data.gz")
        self.deid_df.to_csv(self.deid_path, index=False)

    def _transform_cloud(self):
        """Deidentify a dataset using Gretel's SaaS API."""
        df = pd.read_csv(self.training_path)
        config = yaml.safe_load(self.tx_config)
        model = self.project.create_model_obj(
            config, data_source=df.head(self.preview_recs)
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
        """Print a markdown-format report of the NER findings from a dataset."""
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
            f"Column count: {report['field_count']} columns\n"
            "\n"
            "Dataset overview\n"
            f"{df.to_markdown(index = False)}\n"
            "\n"
        )
        with open(self.deid_report_path, "w") as fh:
            fh.write(report_content)
        print(report_content)

    def _print_transform_report(self):
        """Print a markdown-format report of data transformations on a dataset."""
        report = self.run_report
        report_content = (
            "Transforms finished.\n"
            f"Processing time: {report['summary'][0]['value']} seconds\n"
            f"Record count: {report['summary'][1]['value']}\n"
            "\n"
            "Columns transformed via field header name\n"
            f"{pd.DataFrame(report['summary'][2]['value']).to_markdown(index = False)}\n"
            "\n"
        )
        with open(self.deid_report_path, "a") as fh:
            fh.write(report_content)
        print(report_content)

    def transform(self):
        """Deidentify a dataset using Gretel's Transform APIs."""

        if self._cache_init_report.exists() and self._cache_run_report.exists():
            self.init_report = pickle.load(open(self._cache_init_report, "rb"))
            self.run_report = pickle.load(open(self._cache_run_report, "rb"))
            self.deid_df = pd.read_csv(self.deid_path)
        else:
            # Initialize transform model
            if self.run_mode == "cloud":
                self._transform_cloud()
            elif self.run_mode == "local":
                self._transform_local()

            pickle.dump(self.init_report, open(self._cache_init_report, "wb"))
            pickle.dump(self.run_report, open(self._cache_run_report, "wb"))
            self.deid_df.to_csv(self.deid_path, index=False)

        self._print_ner_report()
        self._print_transform_report()

    def _synthesize_cloud(self):
        """Train a synthetic data model on a dataset using Gretel's SaaS APIs.
        """
        config = read_model_config("synthetics/tabular-actgan")
        config["models"][0]["actgan"]["generate"] = {"num_records": len(self.deid_df)}
        config["models"][0]["actgan"]["params"]["epochs"] = 100
        model = self.project.create_model_obj(
            model_config=config, data_source=self.deid_path
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

    def _synthesize_local(self):
        """Train a synthetic data model on a dataset using Gretel's Hybrid Cloud APIs."""
        pass

    def synthesize(self):
        """Train a synthetic data model on a dataset and use it to create an artificial
        version of a dataset with increased privacy guarantees.
        """
        if self.run_mode == "cloud":
            self._synthesize_cloud()
        elif self.run_mode == "local":
            self._synthesize_local()
