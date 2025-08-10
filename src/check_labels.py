# check_labels.py

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse

def find_suspicious_labels(data_path, k=5, threshold=0.5):
    """
    Analyzes a dataset to find rows with potentially flipped labels using KNN.

    A row is considered suspicious if its label disagrees with a certain threshold
    of its k-nearest neighbors.

    Args:
        data_path (str): Path to the CSV data file.
        k (int): Number of neighbors to consider.
        threshold (float): Fraction of neighbors that must disagree to flag a point (e.g., 0.5 means 50% or more).

    Returns:
        list: A list of indices for rows with suspicious labels.
    """
    print(f"Checking for suspicious labels in: {data_path}")
    print(f"Using k={k} and threshold={threshold}\n")
    
    # Load the data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['species'])
    y = df['species']

    # We use KNeighborsClassifier as a convenient way to find neighbors.
    # We ask for k+1 neighbors because the closest neighbor to any point is the point itself.
    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(X, y)

    # Find the k+1 nearest neighbors for every point in the dataset.
    # The 'indices' array will contain the row index of each neighbor.
    distances, indices = knn.kneighbors(X)
    # print(f"dfldjdlj:: {distances} and indics :: {indices}")
    # return

    suspicious_indices = []
    # Iterate through each data point
    for i in range(len(df)):
        original_label = y.iloc[i]
        
        # Get the labels of the k *other* neighbors (exclude the point itself, which is at index 0)
        neighbor_indices = indices[i][1:]
        neighbor_labels = y.iloc[neighbor_indices]
        
        # Count how many neighbors have a different label
        num_mismatched = np.sum(neighbor_labels != original_label)
        
        # Check if the mismatch ratio exceeds our threshold
        if (num_mismatched / k) >= threshold:
            suspicious_indices.append(i)
            # Optional: Print details for each suspicious point found
            # print(f"  - Row {i} is suspicious. Label is '{original_label}', but {num_mismatched}/{k} neighbors disagree.")

    print(f"\n--- Report ---")
    print(f"Found {len(suspicious_indices)} suspicious labels out of {len(df)} total rows.")
    if suspicious_indices:
        print(f"Suspicious row indices: {suspicious_indices}")
    print("--------------")
    
    return suspicious_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for suspicious labels in a dataset using KNN.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the input CSV file to check.")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to check against.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fraction of neighbors that must disagree to flag a point.")
    
    args = parser.parse_args()

    find_suspicious_labels(data_path=args.data_path, k=args.k, threshold=args.threshold)
