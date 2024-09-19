import sys
sys.path.append('../../')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    # Load the processed dataset
    df = store.get_processed("transformed_dataset.csv")
    logging.info("Loaded processed dataset.")
    print("3")
    print(df)
    # Perform train-test split
    df_train, df_test = train_test_split(df, test_size=config.get("test_size", 0.2), random_state=config.get("random_state", 42))
    
    # Initialize the RandomForestClassifier with the parameters from the config
    rf_estimator = RandomForestClassifier(**config["random_forest"])
    
    # Initialize the SklearnClassifier with the estimator, features, and target
    model = SklearnClassifier(rf_estimator, config["features"], config["target"])
    
    # Train the model
    model.train(df_train)

    # Evaluate the model
    metrics = model.evaluate(df_test)

    # Save the model and metrics
    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)

    # Optional: Print out the metrics
    print("Model Evaluation Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
