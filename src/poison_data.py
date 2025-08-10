import pandas as pd
import numpy as np
import argparse
import os
import random

def poison_labels(input_path, output_path, poison_level):
    """
    Loads a CSV, flips the labels for a specified percentage of rows,
    and saves the result.

    Args:
        input_path (str): Path to the original CSV file.
        output_path (str): Path to save the poisoned CSV file.
        poison_level (float): The percentage of labels to flip (e.g., 0.10 for 10%).
    """
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Get the unique labels in the target column
    unique_labels = df['species'].unique().tolist()
    if len(unique_labels) < 2:
        print("Error: Cannot flip labels with less than two unique classes.")
        return

    # Determine the number of rows to poison
    num_rows = len(df)
    num_to_poison = int(num_rows * poison_level)

    if num_to_poison == 0 and poison_level > 0:
        print(f"Warning: Poison level {poison_level * 100}% is too low to select any rows. No labels will be flipped.")
        df.to_csv(output_path, index=False)
        return

    print(f"Flipping labels for {num_to_poison} of {num_rows} rows ({poison_level * 100:.2f}%)...")
    
    # Create a copy to modify
    df_poisoned = df.copy()

    # Select random row indices to poison without replacement
    poison_indices = np.random.choice(df.index, size=num_to_poison, replace=False)

    # Flip the label for each selected row
    for idx in poison_indices:
        original_label = df_poisoned.loc[idx, 'target']
        # Create a list of possible new labels (all labels except the original one)
        possible_new_labels = [label for label in unique_labels if label != original_label]
        # Choose a new label randomly from the possibilities
        new_label = random.choice(possible_new_labels)
        # Apply the flipped label
        df_poisoned.loc[idx, 'target'] = new_label
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_poisoned.to_csv(output_path, index=False)
    print(f"Poisoned data with flipped labels saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a dataset by flipping labels.")
    parser.add_argument("--input-path", type=str, default="data/iris.csv", help="Path to the input CSV file.")
    parser.add_argument("--output-path", type=str, default="data/iris_poisoned.csv", help="Path for the poisoned output CSV.")
    parser.add_argument("--poison-level", type=float, required=True, help="Fraction of labels to flip (e.g., 0.05 for 5%).")
    
    args = parser.parse_args()

    if not 0.0 <= args.poison_level <= 1.0:
        raise ValueError("Poison level must be between 0.0 and 1.0")

    poison_labels(args.input_path, args.output_path, args.poison_level)