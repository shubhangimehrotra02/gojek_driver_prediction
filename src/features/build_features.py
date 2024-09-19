import sys
sys.path.append('../../')

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.transformations import (
    temporal_features,
    participant_general_specific_features,
    driver_distance_to_pickup,
    geographical_features
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()
    dataset = store.get_processed("dataset.csv")
    dataset = dataset[dataset["participant_status"] != "CREATED"]
    dataset = apply_feature_engineering(dataset)
    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print(df)
    return (
        df.pipe(temporal_features)
          .pipe(participant_general_specific_features)
          .pipe(driver_distance_to_pickup)
          .pipe(geographical_features)
    )


if __name__ == "__main__":
    main()