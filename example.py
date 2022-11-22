import json
import pandas as pd
from smart_open import open

from anonymizer import Anonymizer

def preprocess_data(ds: str) -> str:
    """Fill NaN values with "" if they exist in the input data
    as required by ACTGAN
    """
    output_path = 'training_data.csv'
    df = pd.read_csv(ds)
    df = df.fillna("")
    df.to_csv(output_path, index=False)
    return output_path

def main():
    dataset_path = "./data/google-meet-mycompany.csv"
    training_path = preprocess_data(dataset_path)

    am = Anonymizer(run_mode="cloud")
    am.anonymize(dataset_path=training_path)


if __name__ == "__main__":
    main()
