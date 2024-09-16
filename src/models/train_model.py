from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def calculate_vif(df, features):
    """
    Calculates the VIF for each feature in the dataset.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    # Load the processed dataset
    df = store.get_processed("transformed_dataset.csv")

    # Calculate scale_pos_weight for class imbalance handling
    positive_class_samples = df[df[config["target"]] == 1].shape[0]
    negative_class_samples = df[df[config["target"]] == 0].shape[0]
    scale_pos_weight = negative_class_samples / positive_class_samples
    print(f"Scale Pos Weight: {scale_pos_weight}")

    # Split the dataset into training and test sets
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    # Extract features (X) and target (y) from the training dataset
    X_train = df_train[config["features"]]
    y_train = df_train[config["target"]]

    # Calculate VIF to check for multicollinearity
    print("Checking for multicollinearity using VIF...")
    vif_data = calculate_vif(df_train, config["features"])
    print(vif_data)

    # Define the XGBoost model with parameters from config
    xgb_estimator = XGBClassifier(
        n_estimators=config["xgboost"]["n_estimators"],
        max_depth=config["xgboost"]["max_depth"],
        learning_rate=config["xgboost"]["learning_rate"],
        subsample=config["xgboost"]["subsample"],
        colsample_bytree=config["xgboost"]["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        random_state=config["xgboost"]["random_state"]
    )

    # Perform cross-validation on the raw XGBoost estimator
    scores = cross_val_score(xgb_estimator, X_train, y_train, cv=5, scoring='f1')
    print("F1 Score for each fold:", scores)
    print("Average F1 Score:", scores.mean())

    # Initialize the SklearnClassifier wrapper with XGBoost model
    model = SklearnClassifier(xgb_estimator, config["features"], config["target"])

    # Train the model using the SklearnClassifier wrapper
    model.train(df_train)

    # Evaluate the model on the test set
    metrics = model.evaluate(df_test)

    # Save the trained model and evaluation metrics
    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)

    # Get feature importance from the XGBoost model
    booster = xgb_estimator.get_booster()
    importance = booster.get_score(importance_type='weight')

    # Print the feature importance
    print("Feature Importance:", importance)


if __name__ == "__main__":
    main()
