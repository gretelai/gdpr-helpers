import glob

from gdpr_helpers import Anonymizer

search_pattern = "data/*.csv"


def main():
    am = Anonymizer(
        project_name="gdpr-workflow", 
        run_mode="cloud", 
        overwrite=False,
        )

    for dataset_path in glob.glob(search_pattern):
        am.anonymize(dataset_path=dataset_path)


if __name__ == "__main__":
    main()
