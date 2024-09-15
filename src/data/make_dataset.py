import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.utils.store import AssignmentStore

def main():
    store = AssignmentStore()
    config = load_config()

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

    # Add new feature: driver_distance
    dataset = add_driver_distance(dataset)

    # Add new features: is_rush_hour and distance_ratio
    dataset = add_rush_hour_feature(dataset)
    dataset = add_distance_ratio(dataset)

    # Save the processed training dataset
    store.put_processed("dataset.csv", dataset)
    print("Processed training dataset saved:", dataset.shape)

def process_test_data(store):
    # Load test data
    df_test = store.get_raw("test_data.csv")
    print("Test data loaded:", df_test.shape)

    # Add new feature: driver_distance
    df_test = add_driver_distance(df_test)

    # Add new features: is_rush_hour and distance_ratio
    df_test = add_rush_hour_feature(df_test)
    df_test = add_distance_ratio(df_test)

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
    df['event_hour'] = pd.to_datetime(df['event_timestamp']).dt.hour
    df['is_rush_hour'] = df['event_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    return df

# Add distance ratio feature
def add_distance_ratio(df):
    df['distance_ratio'] = df['trip_distance'] / (df['driver_distance'] + 1)  # Avoid division by zero
    return df

if __name__ == "__main__":
    main()
