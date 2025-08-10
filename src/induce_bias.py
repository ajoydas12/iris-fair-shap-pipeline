# src/induce_bias.py

import pandas as pd
import numpy as np
import os

def induce_bias(data_path):
    """
    Loads the Iris dataset and adds a new 'location' column with
    intentionally biased values to test fairness metrics.

    In this scenario, 'virginica' flowers are made to be much more
    prevalent in 'location' 1.

    Args:
        data_path (str): The path to the iris.csv file.
    """
    print(f"--- Inducing bias in data at: {data_path} ---")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # Use a reproducible random seed
    np.random.seed(42)
    
    locations = []
    for species in df['species']:
        if species == 'Virginica':
            # 80% chance of being in location 1 if the species is virginica
            locations.append(np.random.choice([0, 1], p=[0.2, 0.8]))
        else:
            # 80% chance of being in location 0 for other species
            locations.append(np.random.choice([0, 1], p=[0.8, 0.2]))
            
    df['location'] = locations
    # print(f"DLAJ L :: {locations} and length ::{len(locations)}")
    # Save the modified DataFrame back to the same file
    df.to_csv(data_path, index=False)
    
    print("Successfully added biased 'location' column to the dataset.")
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    induce_bias("data/iris.csv")