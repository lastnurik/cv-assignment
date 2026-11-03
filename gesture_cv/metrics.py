"""Auditable classification metrics from aligned, saved per-sample predictions."""
import math
from collections import defaultdict
from . import CLASS_NAMES

def classification_metrics(rows):
    if not rows:raise ValueError('No predictions')
    n=len(CLASS_NAMES);cm=[[0]*n for _ in range(n)];ids=set()
    for row in rows:
        if row['sample_id'] in ids:raise ValueError('Duplicate prediction ID')
        ids.add(row['sample_id'])
        truth=CLASS_NAMES.index(row['label']);pred=CLASS_NAMES.index(row['prediction']);cm[truth][pred]+=1
    classes={}
    for i,c in enumerate(CLASS_NAMES):
        tp=cm[i][i];support=sum(cm[i]);predicted=sum(r[i] for r in cm)
        precision=tp/predicted if predicted else 0.0;recall=tp/support if support else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        classes[c]={'precision':precision,'recall':recall,'f1':f1,'support':support}
    return {'accuracy':sum(cm[i][i] for i in range(n))/len(rows),
            'macro_f1':sum(x['f1'] for x in classes.values())/n,'per_class':classes,
            'confusion_matrix':cm,'class_order':list(CLASS_NAMES),'samples':len(rows),'zero_division':0}

def evaluation_report(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row.get('environment','unknown')].append(row)
    return {'overall':classification_metrics(rows),'by_environment':{e:classification_metrics(r) for e,r in sorted(grouped.items())}}

def compare_aligned(runs):
    if len(runs)<2:raise ValueError('Need at least two runs')
    reference=None
    for name,rows in runs.items():
        mapping={r['sample_id']:(r['label'],r.get('environment','unknown')) for r in rows}
        if len(mapping)!=len(rows):raise ValueError(f'Duplicate IDs in {name}')
        if reference is None:reference=mapping
        elif mapping!=reference:raise ValueError('Run sample IDs/labels/environments do not match')
    return {name:evaluation_report(rows) for name,rows in runs.items()}
