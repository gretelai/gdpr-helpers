import json
import pickle
from pathlib import Path
from shutil import rmtree
from typing import Optional

import pandas as pd
from gretel_client import configure_session, submit_docker_local
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config
from pkg_resources import resource_filename
from smart_open import open

from gdpr_helpers import reports
from gdpr_helpers.helpers import quiet_poll

PREVIEW_RECS = 100


class Anonymizer:
    """Automated model training and synthetic data generation tool
    Args:
        project_name: Gretel project name. Defaults to "gdpr-anonymized".
        transforms_config: Location of transform config. This can be a local path or URL that
            will be accessible when running.
        synthetics_config: Location of synthetics config. This can be a local path or URL that
            will be accessible when running.
        run_mode: One of ["cloud", "hybrid"].
        show_real_data: Whether to preview real data in the report, which may contain PII. Defaults to `True`.
        output_dir: Directory to store anonymized data and reports to. Defaults to `artifacts`.
        preview_recs: Number of records to use for transforms training. Defaults to 100.
        overwrite: Set to `True` to automatically overwrite previous anonymization checkpoints. Defaults to `False`.
    """

    def __init__(
        self,
        project_name: str = "gdpr-anonymized",
        transforms_config: str = resource_filename(
            __name__, "../config/transforms_config.yaml"
        ),
        synthetics_config: str = resource_filename(
            __name__, "../config/synthetics_config.yaml"
        ),
        run_mode: str = "cloud",
        preview_recs: int = PREVIEW_RECS,
        show_real_data: bool = True,
        output_dir: str = "artifacts",
        tmp_dir: str = "tmp",
        overwrite: bool = False,
        endpoint: str = None,
    ):

        self.project_name = project_name
        self.synthetics_config = synthetics_config
        self.transforms_config = transforms_config
        self.run_mode = run_mode
        self.preview_recs = preview_recs
        self.show_real_data = show_real_data
        self.output_dir = Path(output_dir)
        self.tmp_dir = Path(tmp_dir)
        self.anonymization_report_path = None
        self.anonymized_path = None
        self.deidentified_path = None
        self.dataset_path = None
        self.training_path = Path(self.tmp_dir / "training_data.csv")
        self.preview_path = Path(self.tmp_dir / "preview.csv")
        self._cache_ner_report = None
        self._cache_run_report = None
        self._cache_syn_report = None
        self.dataset_path: Optional[Path] = None
        self.deid_df = None
        self.synthetic_df = None
        self.ner_report = {}
        self.run_report = {}
        self.syn_report = {}

        if endpoint:
            configure_session(
                api_key="prompt", cache="yes", validate=True, endpoint=endpoint
            )
        else:
            configure_session(api_key="prompt", cache="yes", validate=True)

        self.project = create_or_get_unique_project(name=project_name)
        print(f"Follow along with model training at: {self.project.get_console_url()}")

        if self.run_mode not in ["cloud", "hybrid"]:
            raise ValueError("run_mode must be either 'cloud' or 'hybrid'")
        if overwrite:
            if self.tmp_dir.exists() and self.tmp_dir.is_dir():
                rmtree(self.tmp_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def anonymize(self, dataset_path: str):
        """Anonymize a dataset to GDPR standards
        using a pipeline of named entity recognition, data transformations,
        and synthetic data model training and generation.

        Args:
            dataset_path (str): Path or URL to CSV
        """
        print(f"Anonymizing '{dataset_path}'")
        self.dataset_path = dataset_path
        self._preprocess_data(dataset_path)
        self.transform()
        self.synthesize()
        self._save_reports(self.anonymization_report_path)

        print("Anonymization complete")
        print(f" -- Synthetic data stored to: {self.anonymized_path}")
        print(f" -- Anonymization report stored to: {self.anonymization_report_path}")

    def _save_reports(self, output_path: Path):
        """Save anonymization reports to a local file in html format"""
        compare_html = reports.compare(
            training_path=self.training_path,
            deidentified_path=self.deidentified_path,
            anonymized_path=self.anonymized_path,
            show_real_data=self.show_real_data,
        )
        r = (
            f"<h1>{self.dataset_path}</h1>"
            f"<p>{reports.get_header()}</p>"
            f"{reports.ner_report(self.ner_report)['html']}"
            f"{reports.transform_report(self.run_report)['html']}"
            f"{reports.synthesis_report(self.syn_report)['html']}"
            f"{compare_html}"
        )
        self.anonymization_report_path.write_text(reports.style_html(r))

    def _preprocess_data(self, ds: str) -> str:
        """Remove NaNs from input data before training model.

        Args:
            ds (str): Path to source dataset
        """
        df = pd.read_csv(ds)
        nan_columns = df.columns[df.isna().any()].tolist()
        print(
            f"Warning: Found NaN values in training data columns: {nan_columns}. Replacing NaN values with ''."
        )
        df = df.fillna("")
        df.to_csv(self.training_path, index=False)

        # Setup output paths
        prefix = Path(ds).stem
        self.anonymization_report_path = Path(
            self.output_dir / f"{prefix}-anonymization_report.html"
        )
        self.anonymized_path = Path(self.output_dir / f"{prefix}-synthetic_data.csv")
        self.deidentified_path = Path(
            self.output_dir / f"{prefix}-transformed_data.csv"
        )
        self._cache_ner_report = Path(self.tmp_dir / f"{prefix}-ner_report.pkl")
        self._cache_run_report = Path(self.tmp_dir / f"{prefix}-run_report.pkl")
        self._cache_syn_report = Path(self.tmp_dir / f"{prefix}-syn_report.pkl")

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
        self.deid_df.to_csv(self.deidentified_path, index=False)

    def _transform_cloud(self, config: dict):
        """Gretel SaaS API."""
        df = pd.read_csv(self.training_path)
        model = self.project.create_model_obj(
            config, data_source=df.head(self.preview_recs)
        )
        model.submit_cloud()
        quiet_poll(model)
        with open(model.get_artifact_link("report_json")) as fh:
            self.ner_report = json.loads(fh.read())

        # Use model to transform records
        rh = model.create_record_handler_obj(data_source=df)
        rh.submit_cloud()
        quiet_poll(rh)
        with open(rh.get_artifact_link("run_report_json")) as fh:
            self.run_report = json.loads(fh.read())
        self.deid_df = pd.read_csv(rh.get_artifact_link("data"), compression="gzip")
        self.deid_df.to_csv(self.deidentified_path, index=False)

    def transform(self):
        """Deidentify a dataset using Gretel's Transform APIs."""
        config = read_model_config(self.transforms_config)

        if self._cache_ner_report.exists() and self._cache_run_report.exists():
            self.ner_report = pickle.load(open(self._cache_ner_report, "rb"))
            self.run_report = pickle.load(open(self._cache_run_report, "rb"))
            self.deid_df = pd.read_csv(self.deidentified_path)
        else:
            # Initialize transform model
            if self.run_mode == "cloud":
                self._transform_cloud(config=config)
            elif self.run_mode == "hybrid":
                self._transform_hybrid(config=config)

            pickle.dump(self.ner_report, open(self._cache_ner_report, "wb"))
            pickle.dump(self.run_report, open(self._cache_run_report, "wb"))
            self.deid_df.to_csv(self.deidentified_path, index=False)

        print(reports.ner_report(self.ner_report)["md"])
        print(reports.transform_report(self.run_report)["md"])

    def synthesize(self):
        """Train a synthetic data model on a dataset and use it to create an artificial
        version of a dataset with increased privacy guarantees.
        """
        config = read_model_config(self.synthetics_config)

        model_config = config["models"][0]
        model_type = next(iter(model_config.keys()))

        model_config[model_type]["generate"] = {"num_records": len(self.deid_df)}
        model_config[model_type]["data_source"] = str(self.deidentified_path)

        if self._cache_syn_report.exists():
            self.syn_report = pickle.load(open(self._cache_syn_report, "rb"))
            self.synthetic_df = pd.read_csv(self.anonymized_path)
        else:
            if self.run_mode == "cloud":
                self._synthesize_cloud(config=config)
            elif self.run_mode == "hybrid":
                self._synthesize_hybrid(config=config)

        print(reports.synthesis_report(self.syn_report)["md"])

    def _synthesize_cloud(self, config: dict):
        """Gretel SaaS APIs."""
        model = self.project.create_model_obj(
            model_config=config, data_source=str(self.deidentified_path)
        )
        model.submit_cloud()
        quiet_poll(model)
        self.synthetic_df = pd.read_csv(
            model.get_artifact_link("data_preview"), compression="gzip"
        )
        self.synthetic_df.to_csv(self.anonymized_path, index=False)
        with open(model.get_artifact_link("report_json")) as fh:
            self.syn_report = json.loads(fh.read())
            self.syn_report.update(model._data["billing_data"])
            pickle.dump(self.syn_report, open(self._cache_syn_report, "wb"))

    def _synthesize_hybrid(self, config: dict):
        """Gretel Hybrid Cloud APIs"""
        model = self.project.create_model_obj(
            model_config=config, data_source=str(self.deidentified_path)
        )
        run = submit_docker_local(model, output_dir=str(self.tmp_dir))
        self.synthetic_df = pd.read_csv(
            self.tmp_dir / "data_preview.gz", compression="gzip"
        )
        self.synthetic_df.to_csv(self.anonymized_path, index=False)
        self.syn_report = json.loads(open(self.tmp_dir / "report_json.json.gz").read())
        self.syn_report.update(model._data["billing_data"])
        pickle.dump(self.syn_report, open(self._cache_syn_report, "wb"))
