import json
import pandas as pd

def load_metrics_classification(report_path=None, metrics_path=None):
    if report_path is None:
        report_path = r'N:\NewFYP\forgery_app\classification_report.json'
    if metrics_path is None:
        metrics_path = r'N:\NewFYP\forgery_app\training_metrics.json'

    with open(report_path, 'r') as f:
        report = json.load(f)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return report, metrics


def load_metrics_segmentation(csv_path=None):
    if csv_path is None:
        csv_path = r'N:\NewFYP\forgery_app\static\result\segmentation\metrics_segmentation.csv'

    try:
        df = pd.read_csv(csv_path)
        seg_metrics = df.iloc[0].round(4).to_dict()
    except Exception as e:
        seg_metrics = {}

    return seg_metrics, {}
