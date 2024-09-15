import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    # Check if 'booking_status' exists
    if 'booking_status' in df.columns:
        # Filter only the completed bookings
        completed_bookings = df[df['booking_status'] == 'COMPLETED']

        # Count the number of completed bookings per driver
        driver_completed_bookings = completed_bookings.groupby('driver_id').size().reset_index(name='historical_completed_bookings')

        # Merge the counts back into the original dataframe
        df = pd.merge(df, driver_completed_bookings, on='driver_id', how='left')

    # If 'historical_completed_bookings' doesn't exist, create it and set to 0
    if 'historical_completed_bookings' not in df.columns:
        df['historical_completed_bookings'] = 0

    # Fill NaN values with 0 (if there were drivers with no completed bookings)
    df['historical_completed_bookings'] = df['historical_completed_bookings'].fillna(0)

    return df