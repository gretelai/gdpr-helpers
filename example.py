import glob
from anonymizer import Anonymizer

search_pattern = 'data/*.csv'

def main():
    am = Anonymizer(
        run_mode="cloud",
        tx_config="config/transform_config.yaml",
        sx_config="config/synthetics_config.yaml",
        )

    for dataset_path in glob.glob(search_pattern):
        am.anonymize(dataset_path=dataset_path)
        break

if __name__ == "__main__":
    main()
