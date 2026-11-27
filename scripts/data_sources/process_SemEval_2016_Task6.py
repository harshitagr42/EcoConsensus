# SemEval-2016 Task 6: drop the Opinion towards / Sentiment columns and renumber the IDs.
# Copied from data/raw/original/SemEval_2016_Task6/ (data/ is git-ignored; the dataset itself is not redistributed).
# The input is the climate-target file ("Climate Change is a Real Concern", 564 tweets, tab-separated). In this
# project it lives in data/raw/original/SemEval_2016_Task6/; adjust the two paths below to that folder.

import pandas as pd

# Define your file paths here (replace placeholders with your actual paths)
input_file_path = "data/raw/SemEval_2016_Task6/unprocessed.csv"
output_file_path = "data/raw/SemEval_2016_Task6/final_processed.csv"

# Specify the columns you want to remove
columns_to_remove = ["Opinion towards", "Sentiment"]

try:
    # Load tab-separated data
    df = pd.read_csv(
        input_file_path, sep="\t", engine="python", on_bad_lines="skip"
    )

    # Drop the specified columns
    df = df.drop(columns=columns_to_remove, errors="ignore")

    # Replace/Create the ID column to start from 1 sequentially
    df["ID"] = range(1, len(df) + 1)

    # Save the modified DataFrame to a clean CSV file
    df.to_csv(output_file_path, index=False)

    print(f"Success! Cleaned file saved to: {output_file_path}")
    print(f"Remaining columns: {list(df.columns)}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")