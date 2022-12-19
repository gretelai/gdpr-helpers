# Quickstart

## 1. Set up your virtual environment
```shell
# checkout the repo
git clone git@github.com:gretelai/gdpr-helpers.git

# create venv
cd gdpr-helpers
python -m venv ./venv
source venv/bin/activate

# install gdpr-helpers package
pip install -U -e .
```

## 2. Add your [Gretel API](https://console.gretel.cloud) key via the Gretel CLI.
Use the Gretel client to store your API key to disk. This step is optional, the gdpr-helpers will prompt you for an API key if one cannot be found.
```bash
gretel configure
```

## 3. Anonymize a set of sample files
Use the example function to anonymize all datasets in a directory.
```bash
python -m examples.anonymize_files
```
