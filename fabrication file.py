# fabricate_balanced.py
import pandas as pd
import numpy as np
import random
import sys
from pathlib import Path

INPUT = "synthetic_dataset.csv"
OUTPUT = "synthetic_dataset_balanced.csv"

def load_df(path):
    df = pd.read_csv(path)
    # normalize label name
    if 'isfraud' not in df.columns:
        for alt in ['is_fraud','isFraud','fraud','label']:
            if alt in df.columns:
                df['isfraud'] = df[alt]
                break
    if 'isfraud' not in df.columns:
        raise SystemExit("ERROR: No label column found. Please include one named isfraud/is_fraud/isFraud/fraud/label.")
    df['isfraud'] = df['isfraud'].astype(int)
    return df

def fabricate(df):
    counts = df['isfraud'].value_counts()
    if len(counts) == 1:
        raise SystemExit("ERROR: Only one class present; cannot fabricate the other without domain rules.")
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    major_count = counts.max()
    minor_count = counts.min()
    needed = major_count - minor_count
    print("Original counts:", counts.to_dict())
    print(f"Majority class: {majority_class} ({major_count}); Minority class: {minority_class} ({minor_count})")
    if needed <= 0:
        print("Already balanced or minority larger. Saving a copy.")
        df.to_csv(OUTPUT, index=False)
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'isfraud']
    cat_cols = [c for c in df.columns if c not in numeric_cols + ['isfraud']]

    minority_df = df[df['isfraud'] == minority_class].copy()
    if len(minority_df) == 0:
        raise SystemExit("ERROR: No rows for minority class to sample from.")

    # sample with replacement and perturb numericals
    idxs = np.random.choice(minority_df.index, size=needed, replace=True)
    synth = minority_df.loc[idxs].copy().reset_index(drop=True)

    # perturb numeric columns slightly (5% std)
    for col in numeric_cols:
        if synth[col].dtype.kind in 'biufc':
            std = minority_df[col].std()
            scale = 0.05 * (std if std and not np.isnan(std) else 1.0)
            noise = np.random.normal(0, scale, size=len(synth))
            # handle ints vs floats
            if np.issubdtype(minority_df[col].dtype, np.integer):
                synth[col] = (synth[col] + noise).round().astype(int)
                synth[col] = synth[col].clip(lower=0)
            else:
                synth[col] = synth[col] + noise
                # clip negatives where original non-negative
                if (minority_df[col] >= 0).all():
                    synth[col] = synth[col].clip(lower=0)

    # for categorical columns: keep same most of the time, randomly swap 15%
    for col in cat_cols:
        unique_vals = minority_df[col].dropna().unique()
        if len(unique_vals) == 0:
            continue
        mask = np.random.rand(len(synth)) < 0.15
        synth.loc[mask, col] = np.random.choice(unique_vals, size=mask.sum())

    # set label to minority class (usually 1)
    synth['isfraud'] = minority_class

    df_balanced = pd.concat([df, synth], ignore_index=True)
    print("New counts after fabrication:", df_balanced['isfraud'].value_counts().to_dict())
    df_balanced.to_csv(OUTPUT, index=False)
    print(f"Balanced dataset saved to {OUTPUT}")

if __name__ == "__main__":
    df = load_df(INPUT)
    fabricate(df)
