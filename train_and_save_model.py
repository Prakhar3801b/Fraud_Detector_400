# train_and_save_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from utils import load_and_clean
import pandas as pd

DATA = "synthetic_dataset_balanced.csv"  # created from fabricate_balanced.py
MODEL_OUT = "model.pkl"

X, y = load_and_clean(DATA)

# keep an untouched test set from original real distribution: if you have original test data, prefer that.
# Here we do a normal split but you must validate on realistic distribution in production.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print("Training size:", X_train.shape, "Test size:", X_test.shape)

# Choose RandomForest (fast to train, robust, supports probability predict_proba)
model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

# Save model
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to {MODEL_OUT}")
