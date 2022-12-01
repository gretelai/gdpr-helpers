from pathlib import Path
import pandas as pd

def ner_report(report: dict) -> str:
    """Save a markdown-format report of the NER findings from a dataset."""
    df = pd.DataFrame(report["metadata"]["fields"])[
        ["name", "count", "approx_distinct_count", "missing_count", "labels"]
    ]
    df["labels"] = df["labels"].astype(str).replace("[\[\]']", "", regex=True)
    df.rename(
        columns={"labels": "entities_detected", "name": "column_name"}, inplace=True
    )
    content = (
        "\n\nNamed Entity Recognition (NER) finished.\n"
        f"Processing time: {report['training_time_seconds']} seconds\n"
        f"Record count: {report['record_count']} records\n"
        f"Column count: {report['field_count']} columns\n\n"
        f"Dataset overview\n"
        f"{df.to_markdown(index = False)}\n\n"
    )
    return content

def transform_report(report: dict) -> str:
    """Save a markdown-format report of data transformations on a dataset."""
    for item in report['summary']:
        if item['field'] == 'field_transforms':
            df = pd.DataFrame(item['value']) 

    content = (
        "Transforms finished.\n"
        f"Processing time: {report['summary'][0]['value']} seconds\n"
        f"Record count: {report['summary'][1]['value']}\n\n"
        f"Columns transformed\n"
        f"{df.to_markdown(index = False)}\n\n"
    )
    return content

def synthesis_report(report:dict) -> str:
    """Save a markdown-format report of data synthesis on a dataset."""
    content = (
        f"\n\nSynthesis finished.\n"
        f"Lines memorized: {report['memorized_lines']}\n\n"
        f"Privacy report\n"
        f"{pd.DataFrame.from_dict(report['privacy_protection_level'], orient='index').to_markdown(index = True)}\n\n"
        f"Accuracy report\n"
        f"{pd.DataFrame(report['summary']).to_markdown(index = False)}\n\n"
    )
    return content
