# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:40:34 2024

@author: revel
"""

import os
import pandas as pd


def preprocess_fly_data(virgin_file, mated_file, lam_file, x, delay):
    """
    Preprocesses the fly data files according to the specified requirements.

    Args:
        virgin_file (str): The name of the virgin flies CSV file.
        mated_file (str): The name of the mated flies CSV file.
        lam_file (str): The name of the LAM CSV file.
        x (int): The column number to remove.
        delay (int): The delay value used for row removal.

    Returns:
        dict: A dictionary containing the processed dataframes.
    """

    # Load the virgin and mated flies data
    virgin_df = pd.read_csv(virgin_file, skiprows=((2 * delay)+1), header=None)
    mated_df = pd.read_csv(mated_file, skiprows=((2 * delay)+1), header=None)

    # Drop the first and last columns, and exclude the last row
    virgin_df = virgin_df.iloc[:-1, 1:-1]
    mated_df = mated_df.iloc[:-1, 1:-1]

    # Drop the specified column
    virgin_df = virgin_df.drop(virgin_df.columns[x], axis=1)

    # Print the number of remaining rows
    print(f"Virgin flies: {len(virgin_df)} rows remaining")
    print(f"Mated flies: {len(mated_df)} rows remaining")

    # Load the LAM data
    lam_df = pd.read_csv(lam_file, skiprows=1, header=None)

    # Drop the specified column
    lam_df = lam_df.drop(lam_df.columns[x - 1], axis=1)

    # Separate the columns into two groups
    midpoint = (len(lam_df.columns) - 1) // 2 
    first_half_cols = lam_df.columns[:midpoint]
    second_half_cols = lam_df.columns[midpoint:]

    lam_first_half_df = lam_df[first_half_cols]
    lam_second_half_df = lam_df[second_half_cols]

    # Print the number of remaining rows
    print(f"LAM: {len(lam_df)} rows remaining")

    # Store the dataframes in a dictionary
    data = {
        "virgin": virgin_df,
        "mated": mated_df,
        "lam_first_half": lam_first_half_df,
        "lam_second_half": lam_second_half_df
    }

    return data


def calculate_mai_per_night(df, night_interval=48, total_nights=3):
    """
    Calculates the Morning Anticipation Index (MAI) for each column in a dataframe,
    considering the specified night intervals.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        night_interval (int): The number of rows representing one night.

    Returns:
        pd.DataFrame: A new dataframe with the MAI values for each night and column.
    """

    mai_df = pd.DataFrame(index=range(total_nights))

    
    preanticipation_start = 12
    anticipation_start = 18
    night_end = 23


    for col in range(len(df.columns)):
        mai_values = []
        for night in range(total_nights):
            night_shift = night * night_interval
            numerator_start = anticipation_start + night_shift
            numerator_end = night_end + night_shift + 1
            denominator_start = preanticipation_start + night_shift
            denominator_end = night_end + night_shift + 1

            if denominator_end > len(df):
                break

            if numerator_end > len(df):
                break  # Stop if we've reached the end of the data
            
            numerator = df.iloc[numerator_start:numerator_end, col].sum()
            denominator = df.iloc[denominator_start:denominator_end, col].sum()
            mai_values.append(numerator / denominator)
            print(f"mai value added for {col}")

        mai_df.loc[:, col] = mai_values

    return mai_df



expname = "Exp5_LAM_Dmel"

# Define folder paths
base_folder = f"C:/Users/revel/Desktop/Video_work/vw_{expname}_output/"
tracking_folder = os.path.join(base_folder, "tracking_V2/")
plot_folder = os.path.join(base_folder, "plots_V2/")
data_folder = os.path.join(base_folder, "data_V2/")
position_folder = os.path.join(tracking_folder, "combined_tracking_data/")

data = preprocess_fly_data(
    f"{data_folder}30min_virgin_combined.csv", 
    f"{data_folder}30min_mated_combined.csv", 
    f"{position_folder}Exp5_LAM_Dmel_beam.csv", 
    x=4, 
    delay=8
)

# Calculate MAI per night for each dataframe
virgin_mai_df = calculate_mai_per_night(data["virgin"].copy())
mated_mai_df = calculate_mai_per_night(data["mated"].copy())
lam_first_half_mai_df = calculate_mai_per_night(data["lam_first_half"].copy())
lam_second_half_mai_df = calculate_mai_per_night(data["lam_second_half"].copy())

# Combine all dataframes into a single one
all_mai_df = pd.concat([virgin_mai_df, mated_mai_df, lam_first_half_mai_df, lam_second_half_mai_df], axis=1)

# Save all MAI dataframes to a CSV file
all_mai_df.to_csv(f"{base_folder}all_mai_data.csv", index=False)

print("MAI dataframes saved to all_mai_data.csv")