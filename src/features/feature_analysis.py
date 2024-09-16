import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.store import AssignmentStore

def main():
    store = AssignmentStore()

    # Load the processed dataset (assuming dataset.csv exists)
    dataset = store.get_processed("dataset.csv")

    # Select the columns you want to analyze for correlation
    columns_to_analyze = ['trip_distance', 'driver_distance', 'distance_ratio', "event_hour", "driver_gps_accuracy","is_rush_hour"]

    # Calculate the correlation matrix
    corr_matrix = dataset[columns_to_analyze].corr()

    # Print the correlation matrix
    print("Correlation matrix:\n", corr_matrix)

    # Visualize the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()

if __name__ == "__main__":
    main()
