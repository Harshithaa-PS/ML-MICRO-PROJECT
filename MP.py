# ============================================================
# CNC TOOL WEAR PREDICTION - COMPLETE PROJECT
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from google.colab import files

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import GroupKFold, cross_val_predict

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 2. UPLOAD DATASET
# ============================================================

print("Upload FeatureAndMetadata_Milling.csv")

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

print("\nUploaded file:", file_name)

# ============================================================
# 3. LOAD DATASET
# ============================================================

try:
    df_raw = pd.read_csv(file_name, sep=";")
except:
    df_raw = pd.read_csv(file_name)

print("\nInitial dataset shape:", df_raw.shape)

# Some versions of this dataset contain the real column names
# in the first row.
if "CycleToFailureNormalized" not in df_raw.columns:

    first_row = df_raw.iloc[0].astype(str).tolist()

    if "CycleToFailureNormalized" in first_row:

        df = df_raw.iloc[1:].copy()

        df.columns = first_row

        df = df.reset_index(drop=True)

    else:
        df = df_raw.copy()

else:
    df = df_raw.copy()

print("Final dataset shape:", df.shape)

# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

print("\nNumber of columns after cleaning:", df.shape[1])

# ============================================================
# 5. CONVERT DATA TO NUMERIC
# ============================================================

for col in df.columns:

    if col not in ["FileName", "MillingToolType"]:

        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False),
            errors="coerce"
        )

# ============================================================
# 6. CHECK REQUIRED TOOL-LIFE COLUMN
# ============================================================

if "CycleToFailureNormalized" not in df.columns:

    print("\nERROR:")
    print("CycleToFailureNormalized was not found.")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise ValueError(
        "The dataset does not contain "
        "CycleToFailureNormalized."
    )

# ============================================================
# 7. REMOVE ROWS WITHOUT TOOL-LIFE INFORMATION
# ============================================================

df = df.dropna(
    subset=["CycleToFailureNormalized"]
).reset_index(drop=True)

print(
    "\nRows after removing missing tool-life values:",
    len(df)
)

# ============================================================
# 8. CREATE WORN / UNWORN TARGET
# ============================================================

# Project classification rule:
# <= 20% remaining life  -> WORN
# > 20% remaining life   -> UNWORN

threshold = 0.20

df["ToolCondition"] = np.where(
    df["CycleToFailureNormalized"] <= threshold,
    "WORN",
    "UNWORN"
)

df["Target"] = df["ToolCondition"].map({
    "UNWORN": 0,
    "WORN": 1
})

print("\n==============================")
print("TOOL CONDITION DISTRIBUTION")
print("==============================")

print(
    df["ToolCondition"].value_counts()
)

print("\nPercentage:")

print(
    (
        df["ToolCondition"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

# ============================================================
# 9. SELECT SENSOR FEATURES
# ============================================================

# Columns that should NOT be used as ML input
# because they identify the experiment/tool or directly
# contain the target/tool-life information.

exclude_columns = [

    "FileName",

    "NumberOfCycle",
    "SampleIndex",

    "TollIndex",

    "MillingToolType",

    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength",

    "CycleToFailure",
    "CycleToFailureNormalized",

    "ToolCondition",
    "Target"
]

# Keep columns that are not excluded
candidate_columns = [

    col for col in df.columns
    if col not in exclude_columns
]

# Keep numeric columns only
feature_columns = [

    col for col in candidate_columns
    if pd.api.types.is_numeric_dtype(df[col])
]

X = df[feature_columns].copy()

y = df["Target"].copy()

# ============================================================
# 10. REMOVE EMPTY AND CONSTANT FEATURES
# ============================================================

# Remove columns containing no values
X = X.dropna(
    axis=1,
    how="all"
)

# Remove constant columns
X = X.loc[
    :,
    X.nunique(dropna=True) > 1
]

print("\n==============================")
print("FEATURE INFORMATION")
print("==============================")

print(
    "Number of samples:",
    X.shape[0]
)

print(
    "Number of usable sensor features:",
    X.shape[1]
)

# ============================================================
# 11. GROUP INFORMATION
# ============================================================

# Grouping by tool prevents records belonging to the same
# cutting tool from appearing in both training and validation.

if "TollIndex" in df.columns:

    groups = df["TollIndex"].copy()

else:

    # fallback if the column is unavailable
    groups = np.arange(len(df))

# ============================================================
# 12. DEFINE FIVE ML MODELS
# ============================================================

models = {

    "Naive Bayes":
        GaussianNB(),

    "Perceptron":
        Perceptron(
            max_iter=3000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=3,
            random_state=42
        ),

    "SVM":
        SVC(
            kernel="rbf",
            C=2,
            gamma="scale"
        ),

    "MLP":
        MLPClassifier(
            hidden_layer_sizes=(30, 15),
            max_iter=2000,
            random_state=42
        )
}

# ============================================================
# 13. FEATURE SELECTION
# ============================================================

k_features = min(
    30,
    X.shape[1]
)

print(
    "\nFeatures selected for each model:",
    k_features
)

# ============================================================
# 14. CREATE PIPELINE
# ============================================================

def create_pipeline(model):

    return Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=k_features
            )
        ),

        (
            "model",
            model
        )
    ])

# ============================================================
# 15. MODEL EVALUATION
# ============================================================

print("\n==============================")
print("TRAINING AND COMPARING MODELS")
print("==============================")

# Five-fold grouped cross validation
group_cv = GroupKFold(
    n_splits=5
)

results = []

predictions = {}

for name, model in models.items():

    print(
        "\nTraining:",
        name
    )

    pipeline = create_pipeline(model)

    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=group_cv,
        groups=groups
    )

    predictions[name] = y_pred

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1
    })

# ============================================================
# 16. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

# Select best model based on F1-score
results_df = results_df.sort_values(
    by=[
        "F1-Score",
        "Accuracy"
    ],
    ascending=False
).reset_index(
    drop=True
)

results_percent = results_df.copy()

for column in [

    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"

]:

    results_percent[column] = (
        results_percent[column] * 100
    ).round(2)

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

display(
    results_percent
)

# ============================================================
# 17. SELECT BEST MODEL
# ============================================================

best_row = results_df.iloc[0]

best_model_name = best_row["Model"]

print("\n==============================")
print("BEST MODEL")
print("==============================")

print(
    "Model:",
    best_model_name
)

print(
    "Accuracy:",
    round(
        best_row["Accuracy"] * 100,
        2
    ),
    "%"
)

print(
    "Precision:",
    round(
        best_row["Precision"] * 100,
        2
    ),
    "%"
)

print(
    "Recall:",
    round(
        best_row["Recall"] * 100,
        2
    ),
    "%"
)

print(
    "F1-Score:",
    round(
        best_row["F1-Score"] * 100,
        2
    ),
    "%"
)

# ============================================================
# 18. ACCURACY COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    results_percent["Model"],
    results_percent["Accuracy"]
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "CNC Tool Condition - Model Accuracy Comparison"
)

plt.xticks(
    rotation=25
)

plt.ylim(
    0,
    105
)

plt.tight_layout()

plt.show()

# ============================================================
# 19. F1-SCORE COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    results_percent["Model"],
    results_percent["F1-Score"]
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "F1-Score (%)"
)

plt.title(
    "CNC Tool Condition - F1-Score Comparison"
)

plt.xticks(
    rotation=25
)

plt.ylim(
    0,
    105
)

plt.tight_layout()

plt.show()

# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

best_predictions = predictions[
    best_model_name
]

cm = confusion_matrix(
    y,
    best_predictions,
    labels=[0, 1]
)

plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "Confusion Matrix - " +
    best_model_name
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["UNWORN", "WORN"]
)

plt.yticks(
    [0, 1],
    ["UNWORN", "WORN"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()

# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y,
        best_predictions,
        target_names=[
            "UNWORN",
            "WORN"
        ],
        zero_division=0
    )
)

# ============================================================
# 22. TRAIN FINAL MODEL ON COMPLETE DATASET
# ============================================================

best_pipeline = create_pipeline(
    models[best_model_name]
)

best_pipeline.fit(
    X,
    y
)

print(
    "\nFinal model trained successfully."
)

print(
    "Selected model:",
    best_model_name
)

# ============================================================
# 23. GET SELECTED FEATURES
# ============================================================

selector = best_pipeline.named_steps[
    "feature_selection"
]

selected_mask = selector.get_support()

selected_features = X.columns[
    selected_mask
]

print(
    "\nNumber of selected features:",
    len(selected_features)
)

print(
    "\nSelected features:"
)

for feature in selected_features:

    print(
        "-",
        feature
    )

# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

model_bundle = {

    "model":
        best_pipeline,

    "model_name":
        best_model_name,

    "feature_columns":
        X.columns.tolist(),

    "selected_features":
        selected_features.tolist(),

    "threshold":
        threshold
}

model_path = (
    "/content/"
    "cnc_tool_wear_final_model.pkl"
)

joblib.dump(
    model_bundle,
    model_path
)

print(
    "\n=============================="
)

print(
    "FINAL MODEL SAVED"
)

print(
    "File:",
    model_path
)

print(
    "Exists:",
    os.path.exists(
        model_path
    )
)

print(
    "Size:",
    os.path.getsize(
        model_path
    ),
    "bytes"
)

# ============================================================
# 25. DOWNLOAD FINAL MODEL
# ============================================================

files.download(
    model_path
)

# ============================================================
# 26. OPTIONAL NEW DATA TESTING
# ============================================================

print(
    "\n=============================="
)

print(
    "NEW DATA TESTING"
)

print(
    "Upload a NEW CNC CSV file."
)

new_upload = files.upload()

# If user uploads a new file
if len(new_upload) > 0:

    new_file = list(
        new_upload.keys()
    )[0]

    print(
        "\nNew test file:",
        new_file
    )

    # Load test file
    try:

        new_raw = pd.read_csv(
            new_file,
            sep=";"
        )

    except:

        new_raw = pd.read_csv(
            new_file
        )

    # Handle dataset format
    if (
        "CycleToFailureNormalized"
        not in new_raw.columns
    ):

        first_row = (
            new_raw.iloc[0]
            .astype(str)
            .tolist()
        )

        if (
            "CycleToFailureNormalized"
            in first_row
        ):

            new_data = (
                new_raw.iloc[1:]
                .copy()
            )

            new_data.columns = first_row

            new_data = (
                new_data.reset_index(
                    drop=True
                )
            )

        else:

            new_data = new_raw.copy()

    else:

        new_data = new_raw.copy()

    # Clean column names
    new_data.columns = (
        new_data.columns
        .astype(str)
        .str.strip()
    )

    # Convert numeric values
    for col in new_data.columns:

        if col not in [
            "FileName",
            "MillingToolType"
        ]:

            new_data[col] = pd.to_numeric(

                new_data[col]
                .astype(str)
                .str.replace(
                    ",",
                    ".",
                    regex=False
                ),

                errors="coerce"
            )

    # Make sure all training features are present
    missing_features = [

        col

        for col in X.columns

        if col not in new_data.columns
    ]

    print(
        "\nMatching features:",
        len(X.columns)
        - len(missing_features),
        "of",
        len(X.columns)
    )

    if len(missing_features) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Some training features "
            "are missing from the "
            "new CSV."
        )

        print(
            "\nFirst missing features:"
        )

        for col in missing_features[:20]:

            print(
                "-",
                col
            )

    # Reorder features exactly like training
    new_X = new_data.reindex(
        columns=X.columns
    )

    # Predict
    new_prediction = (
        best_pipeline.predict(
            new_X
        )
    )

    # ========================================================
    # 27. FINAL PREDICTION
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "       CNC TOOL CONDITION"
    )

    print(
        "======================================"
    )

    for i, prediction in enumerate(
        new_prediction
    ):

        if prediction == 1:

            condition = "WORN"

            action = (
                "Inspect the cutting tool "
                "and consider maintenance "
                "or replacement."
            )

        else:

            condition = "UNWORN"

            action = (
                "Tool condition is currently "
                "acceptable. Continue monitoring."
            )

        print(
            "\nCNC Sample:",
            i + 1
        )

        print(
            "Prediction:",
            condition
        )

        print(
            "Action:",
            action
        )

    print(
        "\n======================================"
    )

print(
    "\nPROJECT EXECUTION COMPLETED"
)

# ============================================================
# END
# ============================================================
