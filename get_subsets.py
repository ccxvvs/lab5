import pandas as pd
from datasets import load_dataset

# 1. Config
dataset_id = "IrwinLab/LSD_mPro_Fink_2023_noncovalent_Screen2_ZINC22"
print(f"Streaming 1 Million molecules from {dataset_id}...")

# 2. Stream Data
ds = load_dataset(dataset_id, "docking_results", split="train", streaming=True)

# Shuffle with a buffer to get a random mix
shuffled_ds = ds.shuffle(seed=42, buffer_size=10000)

# Take the largest set first (1,000,000)
data_large = list(shuffled_ds.take(1000000))
df_large = pd.DataFrame(data_large)

# 3. Save "Large" (1M)
print(f"Saving large_subset.csv ({len(df_large)} rows)...")
df_large.to_csv("large_subset.csv", index=False)

# 4. Create & Save "Medium" (100k)
print("Subsampling Medium (100k)...")
df_medium = df_large.sample(n=100000, random_state=42)
df_medium.to_csv("medium_subset.csv", index=False)

# 5. Create & Save "Small" (10k)
print("Subsampling Small (10k)...")
df_small = df_medium.sample(n=10000, random_state=42)
df_small.to_csv("small_subset.csv", index=False)

print("Success!")
