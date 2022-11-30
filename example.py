import json

import pandas as pd
from smart_open import open

from anonymizer import Anonymizer


def main():
    dataset_path = "./data/google-meet-mycompany.csv"
    am = Anonymizer(run_mode="cloud")
    am.anonymize(dataset_path=dataset_path)

if __name__ == "__main__":
    main()
