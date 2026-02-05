import pandas as pd
import sklearn
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/subratm62/predictive_maintenance/engine_data.csv"
engine_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = "Engine Condition"

# List of numerical features in the dataset
numeric_features = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]

# Define predictor matrix (X) using selected numeric features
X = engine_dataset[numeric_features]

# Define target variable
y = engine_dataset[target]

# Splitting data into training, validation and test set:
# first we split data into 2 parts, say temporary and test
Xtemp, Xtest, ytemp, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# then we split the temporary set into train and validation

Xtrain, Xval, ytrain, yval = train_test_split(
    Xtemp, ytemp, test_size=0.2, random_state=42, stratify=ytemp
)

print(f"Training Set Size: {Xtrain.shape}\n")
print(f"Validation Set Size:     {Xval.shape}\n")
print(f"Test Set Size:     {Xtest.shape}\n")

# check distribution of two classes for target variable
print(f"Training Set: {ytrain.value_counts(normalize=True)}\n")
print(f"Validation Set: {yval.value_counts(normalize=True)}\n")
print(f"Test Set: {ytest.value_counts(normalize=True)}\n")

Xtrain.to_csv("Xtrain.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytest.to_csv("ytest.csv",index=False)
Xval.to_csv("Xval.csv",index=False)
yval.to_csv("yval.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","Xval.csv","ytrain.csv","yval.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="subratm62/predictive_maintenance",
        repo_type="dataset",
    )
