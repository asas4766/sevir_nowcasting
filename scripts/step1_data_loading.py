"""
STEP 1: SEVIR Data Loading
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
import h5py
import numpy as np

DATA_DIR = "/content/sevir_data"
os.makedirs(DATA_DIR, exist_ok=True)

print("="*60)
print("STEP 1: Downloading SEVIR Catalog")
print("="*60)

# Connect to public S3 bucket
s3 = boto3.client("s3", region_name="us-west-2",
                  config=Config(signature_version=UNSIGNED))
BUCKET = "sevir"

# Download catalog
catalog_path = os.path.join(DATA_DIR, "CATALOG.csv")
if not os.path.exists(catalog_path):
    print("Downloading CATALOG.csv ...")
    s3.download_file(BUCKET, "CATALOG.csv", catalog_path)

# Load and filter
catalog = pd.read_csv(catalog_path, low_memory=False)
vil_catalog = catalog[catalog["img_type"] == "vil"].copy()
vil_catalog = vil_catalog[vil_catalog["file_index"].notnull()]
print(f"Total VIL events: {len(vil_catalog)}")

# Get best shard
shard_counts = vil_catalog["file_name"].value_counts()
best_shard = shard_counts.index[0]
events_in_shard = vil_catalog[vil_catalog["file_name"] == best_shard]

# Select events - DISJOINT SPLIT (no overlap)
selected_events = events_in_shard.sample(n=15, random_state=42).reset_index(drop=True)
train_events = selected_events.iloc[:10]   # rows 0-9
test_events = selected_events.iloc[10:15]  # rows 10-14 (DISJOINT from train)

print(f"\nSelected shard: {best_shard}")
print(f"Train events: {len(train_events)} (rows 0-9)")
print(f"Test events: {len(test_events)} (rows 10-14)")
print(f"Overlap check: {len(set(train_events.index) & set(test_events.index))} overlapping rows")

# Verify no overlap
train_ids = set(train_events['id'].values)
test_ids = set(test_events['id'].values)
overlap = train_ids & test_ids
print(f"Unique event IDs - Train: {len(train_ids)}, Test: {len(test_ids)}")
print(f"Overlapping IDs: {len(overlap)}")
if overlap:
    print(" WARNING: Overlapping events detected!")
else:
    print("✅ No data leakage - train and test sets are disjoint")

# Download shards with corrected path
print("\n" + "="*60)
print("STEP 2: Downloading shard files")
print("="*60)

def ensure_shard(relative_file_name):
    """Download shard file with corrected S3 path"""
    local_path = os.path.join(DATA_DIR, os.path.basename(relative_file_name))

    if not os.path.exists(local_path):
        # Catalog says 'vil/2018/...' but S3 needs 'data/vil/2018/...'
        s3_key = f"data/{relative_file_name}"

        print(f"\nDownloading {os.path.basename(relative_file_name)}...")
        print(f"S3 key: {s3_key}")
        print("This may take 5-15 minutes...")

        s3.download_file(BUCKET, s3_key, local_path)
        size_gb = os.path.getsize(local_path) / (1024**3)
        print(f"✓ Downloaded! Size: {size_gb:.2f} GB")
    else:
        print(f"✓ Already exists: {os.path.basename(relative_file_name)}")

    return local_path

shard_paths = {}
for shard_name in selected_events["file_name"].unique():
    shard_paths[shard_name] = ensure_shard(shard_name)

# Extract events
print("\n" + "="*60)
print("STEP 3: Extracting events")
print("="*60)

def load_events(events_df, shard_paths):
    arrays, ids = [], []
    for idx, row in events_df.iterrows():
        with h5py.File(shard_paths[row["file_name"]], "r") as hf:
            # Show available datasets on first load
            if idx == 0:
                print(f"\nDatasets in HDF5 file:")
                for key in hf.keys():
                    print(f"  {key}: shape {hf[key].shape}")

            # Load VIL data
            vil = hf["vil"][int(row["file_index"])]
        arrays.append(vil)
        ids.append(row["id"])
        if (idx + 1) % 3 == 0:
            print(f"  Processed {idx + 1}/{len(events_df)} events...")
    return np.stack(arrays), ids

print("Extracting training events...")
train_vil, train_ids = load_events(train_events, shard_paths)

print("\nExtracting test events...")
test_vil, test_ids = load_events(test_events, shard_paths)

# Save compact version
output_path = os.path.join(DATA_DIR, "sevir_vil_subset.npz")
np.savez_compressed(
    output_path,
    train_vil=train_vil,
    train_ids=np.array(train_ids),
    test_vil=test_vil,
    test_ids=np.array(test_ids),
)

print(f"\n" + "="*60)
print(" DATA READY!")
print("="*60)
print(f"Train: {train_vil.shape} (10 events)")
print(f"Test: {test_vil.shape} (5 events)")
print(f"Memory: {(train_vil.nbytes + test_vil.nbytes) / 1024**2:.0f} MB")
print(f"Saved: {output_path}")

# Clean up
print("\nCleaning up large shard files...")
for shard_path in shard_paths.values():
    if os.path.exists(shard_path):
        size_gb = os.path.getsize(shard_path) / (1024**3)
        os.remove(shard_path)
        print(f" Deleted {os.path.basename(shard_path)} ({size_gb:.1f} GB freed)")

