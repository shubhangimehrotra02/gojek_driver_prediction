import pandas as pd
from src.utils.store import AssignmentStore
from src.utils.config import load_config
from src.models.classifier import SklearnClassifier

# Add the Haversine formula for distance calculation
import numpy as np

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
    df['driver_distance'] = haversine(
        df['driver_latitude'], df['driver_longitude'], 
        df['pickup_latitude'], df['pickup_longitude']
    )
    return df

# Add rush hour feature
def add_rush_hour_feature(df):
    df['event_hour'] = pd.to_datetime(df['event_timestamp']).dt.hour  # Adjust the column name here
    df['is_rush_hour'] = df['event_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    return df

# Add distance ratio feature
def add_distance_ratio(df):
    df['distance_ratio'] = df['trip_distance'] / (df['driver_distance'] + 1)  # Avoid division by zero
    return df

def main():
    store = AssignmentStore()
    config = load_config()

    df_test = store.get_processed("test_data.csv")
    
    # Ensure test data has the same features as training
    df_test = add_driver_distance(df_test)
    df_test = add_rush_hour_feature(df_test)
    df_test = add_distance_ratio(df_test)

    model = store.get_model("saved_model.pkl")

    df_test["score"] = model.predict(df_test)

    # Save the results
    store.put_processed("results.csv", df_test[["order_id", "driver_id", "score"]])

if __name__ == "__main__":
    main()
