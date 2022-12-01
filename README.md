## Installation
This is how I set this repo up to run the example.

```shell
# checkout the repo
git clone git@github.com:gretelai/gdpr-helpers.git

# create venv
cd gdpr-helpers
python -m venv ./venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Comments/ideas

- The output of the example script is a bit noisy, maybe we could redirect all the output coming from gretel client to a log file and only output stuff from the anonymizer?
  - More advanced option - some kind of `tqdm` progress bar?
- I think that specifying type in docstrings (e.g. `ds (str):`) is no longer necessary with type hints.
- Package structure - if user is supposed to depend on this package and import it, we should put it inside of a module, e.g. `gretel_gdpr` (or `gdpr_helper`) and then put `anonymizer.py` and rest in there.