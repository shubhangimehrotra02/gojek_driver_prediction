import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import logging

from src.features.build_features import apply_feature_engineering
from src.utils.guardrails import validate_prediction_results
from src.utils.store import AssignmentStore


# Add the Haversine formula for distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in kilometers

# Add driver distance feature
def add_driver_distance(df):
    if 'driver_distance' not in df.columns:
        df['driver_distance'] = haversine(
            df['driver_latitude'], df['driver_longitude'], 
            df['pickup_latitude'], df['pickup_longitude']
        )
    return df

# Add rush hour feature
def add_rush_hour_feature(df):
    if 'is_rush_hour' not in df.columns:
        df['event_hour'] = pd.to_datetime(df['event_timestamp']).dt.hour
        df['is_rush_hour'] = df['event_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    return df

# Add distance ratio feature
def add_distance_ratio(df):
    if 'distance_ratio' not in df.columns:
        df['distance_ratio'] = df['trip_distance'] / (df['driver_distance'] + 1)  # Avoid division by zero
    return df

# Ensure all required features are present
def ensure_required_features(df):
    required_features = [
        'hourofday', 'dayofweek', 'hourofday_average', 
        'driver_cluster_label', 'driver_dayofweek_average', 
        'driver_hourofday_average', 'is_completed_last5day_rolling_mean', 
        'dayofweek_average', 'customer_cluster_label'
    ]
    for feature in required_features:
        if feature not in df.columns:
            logging.warning(f"Feature '{feature}' not found in dataframe. Creating placeholder.")
            df[feature] = 0  # Placeholder logic, you may replace this with actual logic
    
    return df

# Apply all feature engineering steps
def apply_feature_engineering(df):
    df = add_driver_distance(df)
    df = add_rush_hour_feature(df)
    df = add_distance_ratio(df)
    df = ensure_required_features(df)  # Ensure that all required features are in the dataframe
    return df

@validate_prediction_results
def main():
    store = AssignmentStore()

    # Load test data
    df_test = store.get_raw("test_data.csv")
    logging.info("Loaded test dataset.")

    # Apply feature engineering to ensure all required features are present
    df_test = apply_feature_engineering(df_test)

    # Load the trained model
    model = store.get_model("saved_model.pkl")
    logging.info("Loaded trained model.")

    # Make predictions
    df_test["score"] = model.predict(df_test)

    # Choose the best driver for each order
    selected_drivers = choose_best_driver(df_test)

    # Save the results
    store.put_predictions("results.csv", selected_drivers)
    logging.info("Saved predictions to results.csv.")


def choose_best_driver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("order_id").agg({"driver_id": list, "score": list}).reset_index()
    df["best_driver"] = df.apply(
        lambda r: r["driver_id"][np.argmax(r["score"])], axis=1
    )
    df = df.drop(["driver_id", "score"], axis=1)
    df = df.rename(columns={"best_driver": "driver_id"})
    return df


if __name__ == "__main__":
    main()
