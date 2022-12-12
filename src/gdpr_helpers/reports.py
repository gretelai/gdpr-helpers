import pandas as pd


def style_html(html: str) -> str:
    template = """
<html>
<head>
<style>

    h2 {
        text-align: left;
        font-family: Helvetica, Arial, sans-serif;
    }
    table { 
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        align: left;
        padding: 5px;
        text-align: left;
        font-family: Helvetica, Arial, sans-serif;
        font-size: 90%;
    }
    table tbody tr:hover {
        background-color: #dddddd;
    }
    .wide {
        width: 90%; 
    }

</style>
</head>
<body>
[[HTML]]   
</body>
</html>
"""
    return template.replace("[[HTML]]", html)


def ner_report(report: dict):
    """Save a markdown-format report of the NER findings from a dataset."""
    df = pd.DataFrame(report["metadata"]["fields"])[
        ["name", "count", "approx_distinct_count", "missing_count", "labels"]
    ]
    df["labels"] = df["labels"].astype(str).replace("[\[\]']", "", regex=True)
    df.rename(
        columns={"labels": "entities_detected", "name": "column_name"}, inplace=True
    )
    md_content = (
        "\n\nNamed Entity Recognition\n"
        f"Processing time: {report['training_time_seconds']} seconds\n"
        f"Samples: {report['record_count']} records\n"
        f"Columns: {report['field_count']} columns\n\n"
        "Entities detected\n"
        f"{df.to_markdown(index = False)}\n\n"
    )
    html_content = (
        "<h2>Named Entity Recognition</h2>"
        f"Processing time: {report['training_time_seconds']} seconds<br>"
        f"Samples: {report['record_count']} records<br>"
        f"Columns: {report['field_count']} columns<br>"
        "<h3>Entities detected</h3>"
        f"<div>{df.to_html(index = False)}</div>"
    )
    return {"md": md_content, "html": html_content}


def transform_report(report: dict):
    """Save a markdown-format report of data transformations on a dataset."""
    for item in report["summary"]:
        if item["field"] == "field_transforms":
            df = pd.DataFrame(item["value"])
    md_content = (
        "Transforms finished.\n"
        f"Processing time: {report['summary'][0]['value']} seconds\n"
        f"Record count: {report['summary'][1]['value']}\n\n"
        "Columns transformed\n"
        f"{df.to_markdown(index = False)}\n\n"
    )
    html_content = (
        "<h2>Transforms</h2>"
        f"Processing time: {report['summary'][0]['value']} seconds<br>"
        f"Record count: {report['summary'][1]['value']}<br>"
        "<h3>Columns transformed</h3>"
        f"<div>{df.to_html(index = False)}</div>"
    )
    return {"md": md_content, "html": html_content}


def synthesis_report(report: dict):
    """Save a markdown-format report of data synthesis on a dataset."""
    summary = pd.DataFrame(report["summary"])
    ppl = pd.DataFrame.from_dict(
        report["privacy_protection_level"], orient="index", columns=["value"]
    )

    md_content = (
        "\n\nSynthesis finished.\n"
        f"Lines memorized: {report['memorized_lines']}\n\n"
        "Privacy report\n"
        f"{ppl.to_markdown(index = True)}\n\n"
        "Accuracy report\n"
        f"{summary.to_markdown(index = False)}\n\n"
    )
    html_content = (
        "<h2>Synthesis</h2>"
        f"Lines memorized: {report['memorized_lines']}<br>"
        "<h3>Privacy report</h3>"
        f"<div>{ppl.to_html(index = True)}</div><br>"
        "<h3>Accuracy report</h3>"
        f"<div>{summary.to_html(index = False)}</div><br>"
    )
    return {"md": md_content, "html": html_content}


def highlight_diff(data, color='lightgreen'):
    highlight = 'background-color: {}'.format(color)
    default = ''
    data = data.fillna('')
    if data[0] != data[1]:
        style = highlight
    else:
        style = default
    return default, style


def compare(
    training_path,
    deidentified_path,
    anonymized_path,
    show_real_data=True,
):
    columns = ["training_data", "deidentified_data", "synthetic_data"]
    tdf = pd.read_csv(training_path).head(1)
    ddf = pd.read_csv(deidentified_path).head(1)
    sdf = pd.read_csv(anonymized_path).head(1)
    cdf = pd.concat([tdf.T, ddf.T, sdf.T], axis=1, keys=columns)
    cdf = cdf.style.apply(highlight_diff, color='#D5F5E3', subset=['training_data', 'deidentified_data'], axis=1).apply(
        highlight_diff, color='#ABEBC6', subset=['deidentified_data', 'synthetic_data'], axis=1)
    
    if not show_real_data:
        cdf = cdf.hide_columns(columns[:1])
    report = ("<h1>Results</h1>" 
              "<p>Highlight values that have been changed between each stage of the anonymization pipeline. Values that have been changed are highlighted in Green.</p>"
              f"<div>{cdf.to_html(index = False)}</div><br>"
    )
    return report
