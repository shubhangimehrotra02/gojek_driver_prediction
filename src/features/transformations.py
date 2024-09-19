import sys
sys.path.append('../../')
import os
import pandas as pd
import numpy as np
import pickle
from haversine import haversine
from IPython.terminal.debugger import set_trace as keyboard
from src.utils.store import AssignmentStore
from sklearn.cluster import KMeans

def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    '''Calculate Haversine distance between driver and pickup locations.'''
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Create temporal features like day of week and is_busy_hour.
    '''
    # Convert event_timestamp to datetime with timezone
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], infer_datetime_format=True).dt.tz_convert('Asia/Kolkata')

    # Extract temporal features
    df["date"] = df["event_timestamp"].dt.date
    df["hourofday"] = df["event_timestamp"].dt.hour
    df["dayofweek"] = df["event_timestamp"].dt.dayofweek
    
    # Mark busy hours
    df["is_busy_hour"] = df["hourofday"].apply(lambda x: 1 if 5 <= x <= 15 else 0)
    
    # Define bins and labels for 'hourofday'
    bins = [0, 6, 12, 18, 24]  # Binning hours into four segments
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']  # Labels for the segments

    # Apply binning to 'hourofday'
    df['hourly_bins'] = pd.cut(df['hourofday'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    return df


def get_cartesian(lat, lon):
    '''
    Convert latitude and longitude to Cartesian coordinates.
    '''
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371  # Radius of the Earth in kilometers
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return [x, y, z]


def geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Use KMeans clustering to extract geographical features from driver and customer locations.
    '''
    # Convert coordinates to Cartesian for clustering
    df_cord_driver = df[["driver_latitude", "driver_longitude"]]
    df_cord_customer = df[["pickup_latitude", "pickup_longitude"]]

    df_cord_driver_cartesian = pd.DataFrame(df_cord_driver.apply(lambda x: get_cartesian(x["driver_latitude"], x["driver_longitude"]), axis=1).tolist(), columns=["x", "y", "z"])
    df_cord_customer_cartesian = pd.DataFrame(df_cord_customer.apply(lambda x: get_cartesian(x["pickup_latitude"], x["pickup_longitude"]), axis=1).tolist(), columns=["x", "y", "z"])

    # Combine driver and customer cartesian coordinates
    combined_coordinates = pd.concat([df_cord_driver_cartesian, df_cord_customer_cartesian])

    # Apply KMeans clustering
    num_clusters = 10  # Define the number of clusters, adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(combined_coordinates)

    # Assign cluster labels back to the dataframe
    df["driver_cluster_label"] = kmeans.predict(df_cord_driver_cartesian)
    df["customer_cluster_label"] = kmeans.predict(df_cord_customer_cartesian)
    
    return df


def participant_general_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Create features capturing variance at both driver and timestamp levels.
    '''
    # Create last 5 days completion rate for each driver
    df['date'] = pd.to_datetime(df['event_timestamp']).dt.date
    df.sort_values(by=['driver_id', 'date'], inplace=True)
    df['is_completed_last5day_rolling_mean'] = df.groupby('driver_id')['is_completed'].rolling(5, min_periods=1).mean().reset_index(drop=True)
    
    # Calculate day of week averages for completion rates
    dayofweek_agg = df.groupby(['driver_id', 'dayofweek']).agg(
        driver_dayofweek_average=('is_completed', 'mean')
    ).reset_index()
    global_dayofweek_agg = df.groupby(['dayofweek']).agg(
        dayofweek_average=('is_completed', 'mean')
    ).reset_index()

    # Merge day of week features
    df = df.merge(dayofweek_agg, on=['driver_id', 'dayofweek'], how='left')\
           .merge(global_dayofweek_agg, on=['dayofweek'], how='left')
    
    # Calculate hour of day averages for completion rates
    hourofday_agg = df.groupby(['driver_id', 'hourofday']).agg(
        driver_hourofday_average=('is_completed', 'mean')
    ).reset_index()
    global_hourofday_agg = df.groupby(['hourofday']).agg(
        hourofday_average=('is_completed', 'mean')
    ).reset_index()

    # Merge hour of day features
    df = df.merge(hourofday_agg, on=['driver_id', 'hourofday'], how='left')\
           .merge(global_hourofday_agg, on=['hourofday'], how='left')

    # Fill missing values with global averages
    df["driver_dayofweek_average"].fillna(df["dayofweek_average"], inplace=True)
    df["driver_hourofday_average"].fillna(df["hourofday_average"], inplace=True)
    df["is_completed_last5day_rolling_mean"].fillna(df["is_completed_last5day_rolling_mean"].mean(), inplace=True)
    
    return df


def participant_mean_acceptance_time(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Create features to understand how quickly or slowly the driver accepts the requests.
    '''
    # Calculate the mean acceptance time for each driver
    df['acceptance_time'] = (df['acceptance_timestamp'] - df['request_timestamp']).dt.total_seconds()
    df['mean_acceptance_time'] = df.groupby('driver_id')['acceptance_time'].transform('mean')
    return df
