# Pol-NLI: download, keep the climate stance rows, rewrite the negated hypothesis and flip its labels.
# Copied from data/raw/original/pol_nli_data/ (data/ is git-ignored; the dataset itself is not redistributed).
# Run inside data/raw/original/pol_nli_data/ (needs `pip install datasets`). The project keeps the output folder
# under the name final_processed/ (renamed by hand from final_processed_pol_nli/). The three-class labels were then
# refined with an LLM; see docs/DATASET.md section 5.2.

import os
from datasets import load_dataset
import pandas as pd

# Define directories
raw_dir = "csv_exports_temp"
stance_dir = "stance_filtered_temp"
final_dir = "final_processed_pol_nli"

# Create output folders
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(stance_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

splits = ["train", "validation", "test"]

print("1. Downloading and exporting mlburnham/Pol_NLI splits to CSV...")
pol_nli = load_dataset("mlburnham/Pol_NLI")
for split_name, split_data in pol_nli.items():
    df = split_data.to_pandas()
    df.to_csv(os.path.join(raw_dir, f"Pol_NLI_{split_name}.csv"), index=False)

print("\n2. Processing and filtering data across splits...")
for split in splits:
    raw_file = os.path.join(raw_dir, f"Pol_NLI_{split}.csv")
    if not os.path.exists(raw_file):
        continue

    df = pd.read_csv(raw_file)

    # Step A: Keep only 'stance detection' task
    df = df[df["task"] == "stance detection"]

    # Step B: Filter specific hypotheses (ignoring case and trailing periods)
    norm_hyp = (
            df["hypothesis"].astype(str).str.strip().str.lower().str.rstrip(".")
    )
    targets = {
            "climate change is a serious concern",
            "climate change is not a concern",
    }
    df = df[norm_hyp.isin(targets)].copy()

    # Re-evaluate normalized hypothesis for the filtered subset
    norm_hyp = (
            df["hypothesis"].astype(str).str.strip().str.lower().str.rstrip(".")
    )

    # Identify rows that were originally "climate change is not a concern"
    neg_mask = norm_hyp == "climate change is not a concern"

    # Step C: Convert those hypothesis texts to "Climate change is a serious concern."
    df.loc[neg_mask, "hypothesis"] = "Climate change is a serious concern."

    # Step D: Flip entailment values ONLY for the rows that were originally negative (0 -> 1, 1 -> 0)
    df.loc[neg_mask, "entailment"] = df.loc[neg_mask, "entailment"].map(
            {0: 1, 1: 0}
    )

    # Step E: Drop 'dataset' and 'task' columns
    df = df.drop(columns=[col for col in ["dataset", "task"] if col in df.columns])

    # Save final processed file
    final_file_path = os.path.join(final_dir, f"Pol_NLI_{split}.csv")
    df.to_csv(final_file_path, index=False)
    print(
            f" - Saved processed '{split}' split ({len(df):,} rows, flipped"
            f" {neg_mask.sum():,} rows) -> {final_file_path}"
    )

print(
        f"\nAll operations complete! Final processed files are saved in:"
        f" '{final_dir}/'"
)


# values of entailment are improved with the help of cluade and gemini ai since they were either 0 or 1 only
# news values 0 = agrees, 1 = neutral, 2 = agains