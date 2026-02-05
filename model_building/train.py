import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import uniform, randint
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report, recall_score, precision_recall_curve, auc
# for model serialization
import joblib
import json
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# -------------------------------
# MLflow Setup
# -------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Predictive-Maintenance")


# -------------------------------
# Load dataset
# -------------------------------
REPO_ID = "subratm62/predictive_maintenance"
REPO_TYPE = "dataset"

def load_csv(filename):
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE
    )
    return pd.read_csv(path)

# Load all datasets
Xtrain = load_csv("Xtrain.csv")
Xtest  = load_csv("Xtest.csv")
Xval  = load_csv("Xval.csv")
ytrain = load_csv("ytrain.csv")
ytest  = load_csv("ytest.csv")
yval  = load_csv("yval.csv")


ytrain = ytrain.to_numpy().ravel()
yval = yval.to_numpy().ravel()
ytest = ytest.to_numpy().ravel()

# -------------------------------
# Features
# -------------------------------
numeric_features = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

# -------------------------------
# Preprocessing
# -------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features)
])

# -------------------------------
# Model
# -------------------------------
gbm = GradientBoostingClassifier(random_state=42)

# Full Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("gbm", gbm)
])

# -------------------------------
# Hyperparameter Search Space
# -------------------------------
param_dist = {
    "gbm__n_estimators": [200, 300, 500, 800],
    "gbm__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "gbm__max_depth": [2, 3, 4],
    "gbm__min_samples_split": [5, 10, 20],
    "gbm__min_samples_leaf": [2, 4, 8],
    "gbm__subsample": [0.6, 0.7, 0.8],
    "gbm__max_features": ["sqrt", "log2", None]
}

f1_macro = make_scorer(f1_score, average="macro")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# -------------------------------
# Hugging Face API
# -------------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "subratm62/predictive-maintenance"
repo_type = "model"

# -------------------------------------
# Experimentation Tracking using mlflow
# -------------------------------------
with mlflow.start_run():

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring=f1_macro,
        cv=cv,
        verbose=2,
        random_state=1,
        n_jobs=-1
    )

    search.fit(Xtrain, ytrain)

    best_model = search.best_estimator_

    # Log every param set
    results = search.cv_results_

    #for i, params in enumerate(results["params"]):
    #    with mlflow.start_run(nested=True):
            # Log hyperparameters
    #        mlflow.log_params(params)
    
            # Log CV metrics
    #        mlflow.log_metric("mean_test_f1_macro", results["mean_test_score"][i])
    #        mlflow.log_metric("std_test_f1_macro", results["std_test_score"][i])
    #        mlflow.log_metric("rank_test_f1_macro", results["rank_test_score"][i])
    
            # Helpful metadata
    #        mlflow.set_tag("trial_number", i + 1)
    #        mlflow.set_tag("cv_folds", cv.get_n_splits())
    #        mlflow.set_tag("scoring_metric", "f1_macro")
            
    # Log best parameters
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_cv_f1_macro", search.best_score_)

    # -------------------------------
    # Threshold Logic
    # -------------------------------
    classification_threshold = 0.50

    def predict_with_threshold(model, X, threshold):
        proba = model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    # -------------------------------
    # Metrics
    # -------------------------------
    y_train_pred = predict_with_threshold(best_model, Xtrain, classification_threshold)
    y_test_pred  = predict_with_threshold(best_model, Xtest, classification_threshold)

    train_report = classification_report(ytrain, y_train_pred, output_dict=True)
    test_report  = classification_report(ytest, y_test_pred, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"]
    })

    # -------------------------------
    # Metadata
    # -------------------------------
    meta = {
        "threshold": classification_threshold,
        "best_params": search.best_params_,
        "cv_f1_macro": search.best_score_,
        "train_metrics": train_report,
        "test_metrics": test_report,
        "features": numeric_features
    }

    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    mlflow.log_artifact("model_meta.json")

    # -------------------------------
    # Save FULL PIPELINE
    # -------------------------------
    model_path = "predictive_maintenance_pipeline.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(best_model, "model")

    # -------------------------------
    # Hugging Face Upload
    # -------------------------------
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type
    )

    api.upload_file(
        path_or_fileobj="model_meta.json",
        path_in_repo="model_meta.json",
        repo_id=repo_id,
        repo_type=repo_type
    )
