from collections import defaultdict
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef
import seaborn as sns
import pandas as pd
import numpy as np
import torch

from src.preprocessing import SemEval


def hamming_score(y_true, y_pred):
    """ from: https://gist.github.com/dschaehi/1ecdb3e53647d62e7daf106266bfa4bc """
    out = ((y_pred & y_true).sum(dim=1) / (y_pred | y_true).sum(dim=1)).mean()
    if out.isnan(): out = torch.tensor(1.0)
    return out

def exact_match_ratio(y_true, y_pred):
    exact_score = []
    for p, r in zip(y_pred, y_true):
        exact_score.append(sorted(p) == sorted(r))
    return sum(exact_score) / len(exact_score)


##### Evaluation Report #####

def eval_report(y_true, y_pred, id_2_label=SemEval.id_2_label):
    ''' Shortcut to report all metrics for a given list of true labels and predicted labels'''

    print(classification_report(y_true, y_pred, target_names=list(id_2_label.values())))
    print("MCC:", matthews_corrcoef(y_true, y_pred))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=list(id_2_label.values())).plot(xticks_rotation="vertical") 


def grouped_eval_report(splits, id_2_label=SemEval.id_2_label):
    outputs = defaultdict(list)   

    for y_true, y_pred in splits:


        for k, v in classification_report(y_true, y_pred, target_names=list(id_2_label.values()), output_dict=True).items():
            
            if isinstance(v, dict): # TODO: Maybe recursive func ...
                for kk, vv in v.items():
                    outputs[f"{k}_{kk}"].append(vv)
            else:
                outputs[k].append(v)


        outputs["mcc"].append(matthews_corrcoef(y_true, y_pred))

    return outputs


def format_grouped_results(splits, target_columns, cell_format="{mean:.3f} ({std:.3f})"):

    headers, rows = [], []
    for k, v in pd.DataFrame.from_dict(grouped_eval_report(splits)).describe()[target_columns].to_dict().items():
        headers.append(k)
        rows.append(cell_format.format(**v))

    print('\t'.join(headers))
    print('\t'.join(rows))


##### Confusion Matrix #####


def grouped_confusion_matrix(splits, id_2_label=SemEval.id_2_label):
    full_true, full_pred = [], []
    for y_true, y_pred in splits:
        full_true += y_true
        full_pred += y_pred

    ConfusionMatrixDisplay(confusion_matrix(full_true, full_pred), display_labels=list(id_2_label.values())).plot(xticks_rotation="vertical")  

##### Loss Graph #####


def extract_epoch_metrics(df, path, loop='train'):
    outputs = []
    for item in df[df[f'{loop}_loss'].isna() == False].to_dict(orient='records'):
        record = {
            'ex_name':path.parts[1],
            'ex_split':path.parts[2].split('_')[-1],
            'epoch':item['epoch'],
            'loop':loop,
        }

        outputs.append({**record, 'value_name':'loss', 'value':item[f'{loop}_loss']})
        outputs.append({**record, 'value_name':'mcc', 'value':item[f'{loop}_mcc']})

    return outputs


def plot_epoch_metrics(target_folder):
    outputs = []

    for path in Path('results').glob(f'{target_folder}/{target_folder}_*/cv_logs/{target_folder}_*/version_0/metrics.csv'):
        df = pd.read_csv(path)
        outputs += extract_epoch_metrics(df, path, loop='train')
        outputs += extract_epoch_metrics(df, path, loop='val')
    
    df = pd.DataFrame.from_dict(outputs)
    g = sns.FacetGrid(data=df, col="ex_name", row='value_name', hue='loop', sharey='row', height=6)
    g.map(sns.lineplot, "epoch", "value")