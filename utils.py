# utils/preprocess.py
import pandas as pd
import numpy as np

def load_and_clean(path):
    df = pd.read_csv(path)
    # normalize label name
    if 'isfraud' not in df.columns:
        for alt in ['is_fraud','isFraud','fraud','label']:
            if alt in df.columns:
                df['isfraud'] = df[alt]
                break
    if 'isfraud' not in df.columns:
        raise ValueError("No label column found. Please include one named isfraud/is_fraud/isFraud/fraud/label.")
    df['isfraud'] = df['isfraud'].astype(int)
    # drop columns that are obviously IDs if present
    for col in ['transaction_id','id','nameOrig','nameDest']:
        if col in df.columns:
            df = df.drop(columns=[col])
    # simple categorical encoding: label-encode small cardinality columns
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype('category').cat.codes
    # fillna with 0 for numerics (or median)
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())
    X = df.drop('isfraud', axis=1)
    y = df['isfraud']
    return X, y
