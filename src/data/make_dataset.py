import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.utils.store import AssignmentStore
import logging


def main():
    store = AssignmentStore()
    config = load_config()

    print(config)

    # Process Training Data
    process_training_data(store, config)

    # Process Test Data
    process_test_data(store)

def process_training_data(store, config):
    # Load booking log and participant log data
    booking_df = store.get_raw("booking_log.csv")
    print("Booking data loaded:", booking_df.shape)

    booking_df = clean_booking_df(booking_df)

    participant_df = store.get_raw("participant_log.csv")
    print("Participant data loaded:", participant_df.shape)

    participant_df = clean_participant_df(participant_df)

    # Merge the datasets
    dataset = merge_dataset(booking_df, participant_df)
    print("Merged dataset:", dataset.shape)

    # Create the target column
    dataset = create_target(dataset, config["target"])

    # Add feature engineering
    dataset = feature_engineering(dataset)

    # Save the processed training dataset
    store.put_processed("dataset.csv", dataset)
    print("Processed training dataset saved:", dataset.shape)

def process_test_data(store):
    # Load test data
    df_test = store.get_raw("test_data.csv")
    print("Test data loaded:", df_test.shape)

    # Feature engineering for test data
    df_test = feature_engineering(df_test)

    # Save the processed test dataset
    store.put_processed("test_data.csv", df_test)
    print("Processed test dataset saved:", df_test.shape)

# The existing utility functions remain the same
def clean_booking_df(df: pd.DataFrame) -> pd.DataFrame:
    unique_columns = [
        "order_id",
        "trip_distance",
        "pickup_latitude",
        "pickup_longitude",
        "booking_status"
    ]
    df = df.drop_duplicates(subset=unique_columns)
    return df[unique_columns]

def clean_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df

def merge_dataset(bookings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(participants, bookings, on="order_id", how="left")
    return df

def create_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[target_col] = df["participant_status"].apply(lambda x: int(x == "ACCEPTED"))
    return df

# Haversine function for distance calculation
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

# Feature Engineering: Adds new features to the dataset
def feature_engineering(df):
    # Add driver distance
    df = add_driver_distance(df)

    # Add rush hour feature
    df = add_rush_hour_feature(df)

    # Add distance ratio
    df = add_distance_ratio(df)

    # Add time of day feature
    df = add_time_of_day(df)

    # Add interaction features
    df['trip_driver_interaction'] = df['trip_distance'] * df['driver_distance']

    return df

# Add driver distance feature
def add_driver_distance(df):
    df['driver_distance'] = haversine(
        df['driver_latitude'], df['driver_longitude'], 
        df['pickup_latitude'], df['pickup_longitude']
    )
    return df

# Add rush hour feature
def add_rush_hour_feature(df):
    df['event_hour'] = pd.to_datetime(df['event_timestamp']).dt.hour
    df['is_rush_hour'] = df['event_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    return df

# Add distance ratio feature
def add_distance_ratio(df):
    df['distance_ratio'] = df['trip_distance'] / (df['driver_distance'] + 1)  # Avoid division by zero
    return df

# Add time of day feature (morning, afternoon, evening, night)
def add_time_of_day(df):
    def time_of_day(hour):
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    df['time_of_day'] = df['event_hour'].apply(time_of_day)
    return df

if __name__ == "__main__":
    main()
