import glob
from gdpr_helpers import Anonymizer

search_pattern = "data/*.csv"


def main():
    am = Anonymizer(
        run_mode="cloud",
        #transforms_config="config/transform_config.yaml",
        #synthetics_config="config/synthetics_config.yaml",
    )

    for dataset_path in glob.glob(search_pattern):
        am.anonymize(dataset_path=dataset_path)


if __name__ == "__main__":
    main()
