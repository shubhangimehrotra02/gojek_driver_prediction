import os
import toml
import json
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Function to calculate VIF (Variance Inflation Factor) for detecting multicollinearity
def calculate_vif(df, features):
    """
    Calculates the VIF for each feature in the dataset.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(df[features].values, i) for i in range(len(features))
    ]
    return vif_data

# Function to find the best threshold based on maximizing the F1 score
def find_best_threshold(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    best_threshold = 0.5
    best_f1 = 0
    for threshold in thresholds:
        y_pred_threshold = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

# Function to ensure the directory exists before writing the file
def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

# Function to train and evaluate multiple models
def run_model_pipeline(
    df_train,
    df_test,
    features,
    target,
    models,
    apply_smote=True,
    apply_undersampling=False,
    apply_adasyn=False,
):
    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    if apply_smote:
        # Apply SMOTE for oversampling
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logging.info("Applied SMOTE for oversampling.")

    if apply_undersampling:
        # Apply Random Undersampling
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        logging.info("Applied Random Undersampling.")

    if apply_adasyn:
        # Apply ADASYN for oversampling
        ada = ADASYN(random_state=42)
        X_train, y_train = ada.fit_resample(X_train, y_train)
        logging.info("Applied ADASYN for oversampling.")

    results = {}

    for model_name, model in models.items():
        try:
            logging.info(f"Training {model_name}...")
            model.fit(X_train, y_train)

            # Optimize threshold based on ROC curve
            best_threshold = find_best_threshold(model, X_test, y_test)
            logging.info(f"Best threshold for {model_name}: {best_threshold}")

            # Evaluate using the best threshold
            y_probs = model.predict_proba(X_test)[:, 1]
            y_pred = (y_probs >= best_threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)
            f1 = f1_score(y_test, y_pred, zero_division=1)

            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "best_threshold": best_threshold,
            }
            logging.info(f"{model_name} metrics: {results[model_name]}")
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")

    return results

@validate_evaluation_metrics
def main():
    try:
        store = AssignmentStore()
        config = load_config()

        # Load the processed dataset
        df = store.get_processed("transformed_dataset.csv")
        logging.info("Loaded processed dataset.")

        features = config["features"]
        target = config["target"]
        test_size = config["test_size"]

        # Extract the XGBoost parameters
        xgb_params = config["xgboost"]

        # Split the dataset into training and test sets using test_size from config
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=xgb_params["random_state"], stratify=df[target]
        )
        logging.info(
            f"Split data into train and test sets with test size = {test_size}."
        )

        # Calculate the scale_pos_weight manually
        positive_class_samples = df_train[df_train[target] == 1].shape[0]
        negative_class_samples = df_train[df_train[target] == 0].shape[0]
        scale_pos_weight = negative_class_samples / positive_class_samples
        logging.info(f"Calculated Scale Pos Weight: {scale_pos_weight}")

        # Checking for multicollinearity
        logging.info("Checking for multicollinearity using VIF...")
        vif_data = calculate_vif(df_train, features)
        logging.info(f"VIF Data:\n{vif_data}")

        # Optionally, you can decide to remove features with high VIF
        high_vif = vif_data[vif_data["VIF"] > 5]
        if not high_vif.empty:
            logging.warning(
                f"Features with high VIF detected:\n{high_vif}. Consider removing or transforming them."
            )
            # Example: Remove features with VIF > 5
            features = [f for f in features if f not in high_vif["feature"].tolist()]
            logging.info(f"Updated feature list after removing high VIF features: {features}")

        # Define the models to be compared with hyperparameter tuning
        models = {
            "XGBoost": XGBClassifier(
                n_estimators=xgb_params["n_estimators"][0],
                max_depth=xgb_params["max_depth"][0],
                learning_rate=xgb_params["learning_rate"][0],
                subsample=xgb_params["subsample"][0],
                colsample_bytree=xgb_params["colsample_bytree"][0],
                scale_pos_weight=scale_pos_weight,  # Use the calculated float value
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=xgb_params["random_state"],
            ),
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", solver="liblinear", random_state=20
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=20
            ),
        }

        # Run the model pipeline and optimize thresholds
        results = run_model_pipeline(
            df_train,
            df_test,
            features,
            target,
            models,
            apply_smote=True,
            apply_undersampling=False,
            apply_adasyn=False,
        )

        # Ensure the directory exists before saving metrics
        metrics_filepath = os.path.join("submission", "metrics.json")
        ensure_directory_exists(metrics_filepath)

        # Save the evaluation results
        store.put_metrics(metrics_filepath, results)
        logging.info(f"Saved evaluation metrics to {metrics_filepath}.")

    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
