"""
===============================================================
COUGH DATASET EXTRACTION + TRAIN/TEST/VALIDATE SPLITTING
===============================================================

BEFORE STARTING:
----------------
We are building a binary cough-classification dataset:

    Original dataset
          |
          v
    Read metadata CSV
          |
          v
    Keep only DRY and WET coughs
          |
          v
    Decide label using the 4 cough-type annotations
          |
          v
    Randomly shuffle the samples
          |
          v
    Split each class:
        90% -> train
         5% -> test
         5% -> validate
          |
          v
    Copy audio files into:

        output/
        ├── train/
        │   ├── dry/
        │   └── wet/
        ├── test/
        │   ├── dry/
        │   └── wet/
        └── validate/
            ├── dry/
            └── wet/

IMPORTANT:
----------
1. The metadata contains 4 cough-type columns because multiple
   annotations can exist for one recording.

2. We use MAJORITY VOTING:
       dry, dry, wet, dry -> dry
       wet, wet, dry, wet -> wet

3. If the annotations are equally divided:
       dry, wet, dry, wet -> SKIP

4. We use a fixed random seed so that the same dataset produces
   the same train/test/validation split every time.

5. Audio files are searched recursively using their UUID.
   Therefore, the original audio dataset can contain subfolders.

6. The code COPIES files. It does not modify or delete the
   original dataset.

REQUIRED:
---------
pip install pandas

Python standard libraries used:
    os
    shutil
    random
    collections

===============================================================
"""

import os
import shutil
import random # To make the splitting of the dataset into train test split - randomly
import pandas as pd
from collections import Counter  # # this is for counting how much time each item occurs in a collection - as here in each audio multiple labelling done (eg: audio 1 - dry dry dry wet)



# =============================================================
# 1. CONFIGURATION
# =============================================================
# Change these paths according to where your dataset is stored.

METADATA_FILE = "C:/Users/aydhi/OneDrive/Desktop/COUGH_CLASSIFIER_PROJECT/DATASET/DRY_WET/public_dataset_v3/metadata_compiled.csv"

# This should be the folder containing your original audio files.
# The code will search inside this folder recursively.
SOURCE_AUDIO_FOLDER = "C:/Users/aydhi/OneDrive/Desktop/COUGH_CLASSIFIER_PROJECT/DATASET/DRY_WET/public_dataset_v3/coughvid_20211012"

# This is where the new dataset will be created.
OUTPUT_FOLDER = "C:/Users/aydhi/OneDrive/Desktop/COUGH_CLASSIFIER_PROJECT/DATASET/CLASSIFIED" 

# Fixed random seed.
# Using the same seed means you can reproduce the same split later.
RANDOM_SEED = 42


# =============================================================
# 2. METADATA COLUMNS
# =============================================================

# Your metadata contains four possible cough-type annotations.
COUGH_TYPE_COLUMNS = [
    "cough_type_1",
    "cough_type_2",
    "cough_type_3",
    "cough_type_4"
]

# Audio extensions that the program will recognize.
# Add another extension here if your dataset uses one not listed.
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".webm",
    ".m4a"
}


# =============================================================
# 3. READ METADATA
# =============================================================

print("\nReading metadata...")

# pandas.read_csv() loads the CSV into a DataFrame.
# A DataFrame is basically a table containing rows and columns.
df = pd.read_csv(METADATA_FILE)

print(f"Total metadata rows: {len(df)}")


# Check that the required columns actually exist - just confirming that the 4 cough type columns and also the uuid columsnd are there in our dataset
required_columns = ["uuid"] + COUGH_TYPE_COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(   # raise ValueError means stop the program and immediately report this error
        f"These required columns are missing from the CSV: "
        f"{missing_columns}")


# =============================================================
# 4. FUNCTION TO DETERMINE DRY/WET LABEL
# =============================================================

def get_cough_label(row):
    """
    Determine the final cough label using majority voting.

    Example:
        dry, dry, wet, dry -> dry
        wet, wet, dry, wet -> wet

    Returns:
        "dry"
        "wet"
        None  -> sample should be skipped
    """

    labels = []

    # Look at the four annotation columns.
    for column in COUGH_TYPE_COLUMNS:

        value = row[column]

        # Ignore missing values.
        if pd.isna(value):
            continue

        # Convert to lowercase so that:
        # "Dry", "DRY", "dry" are treated the same.
        value = str(value).strip().lower()

        # We only care about dry and wet.
        if value in ["dry", "wet"]:
            labels.append(value)

    # No dry/wet annotation available.
    if len(labels) == 0:
        return None

    # Counter counts how many times each label occurs.
    counts = Counter(labels)

    dry_count = counts["dry"]
    wet_count = counts["wet"]

    # Majority vote.
    if dry_count > wet_count:
        return "dry"

    elif wet_count > dry_count:
        return "wet"

    # Equal number of dry and wet annotations.
    # We don't want to randomly assign a wrong label.
    else:
        return None


# Apply our labeling function to every metadata row.
#
# axis=1 means:
# "send one complete row at a time to get_cough_label()"
df["final_label"] = df.apply(get_cough_label, axis=1)


# Remove rows where a final dry/wet label could not be determined.
labeled_df = df[df["final_label"].notna()].copy()

print(f"Dry/Wet samples after labeling: {len(labeled_df)}")


# =============================================================
# 5. SHOW CLASS COUNTS
# =============================================================

print("\nClass distribution:")

class_counts = labeled_df["final_label"].value_counts()

print(f"Dry : {class_counts.get('dry', 0)}")
print(f"Wet : {class_counts.get('wet', 0)}")


# =============================================================
# 6. CREATE OUTPUT FOLDERS
# =============================================================

print("\nCreating output folders...")

for split in ["train", "test", "validate"]:

    for label in ["dry", "wet"]:

        # os.path.join() creates a path correctly for the
        # operating system being used.
        folder = os.path.join(
            OUTPUT_FOLDER,
            split,
            label
        )

        # exist_ok=True means:
        # don't give an error if the folder already exists.
        os.makedirs(folder, exist_ok=True)


# =============================================================
# 7. FIND ALL AUDIO FILES
# =============================================================

print("\nSearching for audio files...")

# Dictionary:
#
#     UUID -> complete audio file path
#
# Example:
#     "abc-123..." -> "dataset/part1/abc-123.wav"
#
# This makes it much faster to find the audio corresponding
# to each UUID in the metadata.
audio_files = {}


# os.walk() recursively visits every folder and subfolder.
for root, directories, files in os.walk(SOURCE_AUDIO_FOLDER):

    for filename in files:

        # Separate filename and extension.
        file_name_without_extension, extension = os.path.splitext(
            filename
        )

        extension = extension.lower()

        # Ignore files that are not audio.
        if extension not in AUDIO_EXTENSIONS:
            continue

        # The dataset uses UUIDs in the metadata.
        # Therefore the audio filename should normally contain
        # the corresponding UUID.
        audio_files[file_name_without_extension] = os.path.join(
            root,
            filename
        )


print(f"Audio files found: {len(audio_files)}")


# =============================================================
# 8. RANDOMLY SPLIT EACH CLASS
# =============================================================

random.seed(RANDOM_SEED)

# We split DRY and WET independently.
#
# This is important because we want approximately the same
# class distribution in train/test/validation.

split_data = {
    "train": [],
    "test": [],
    "validate": []
}


for label in ["dry", "wet"]:

    # Select only one class.
    class_data = labeled_df[
        labeled_df["final_label"] == label
    ].copy()

    # Convert DataFrame rows to a list.
    records = class_data.to_dict("records")

    # Shuffle randomly.
    # Therefore the original order of the metadata does not
    # determine which samples enter train/test/validation.
    random.shuffle(records)

    total = len(records)

    # 5% for test.
    test_count = int(total * 0.05)

    # 5% for validation.
    validate_count = int(total * 0.05)

    # Whatever remains goes into training.
    train_count = total - test_count - validate_count

    # Slice the randomly shuffled list.
    train_records = records[:train_count]

    test_records = records[
        train_count:
        train_count + test_count
    ]

    validate_records = records[
        train_count + test_count:
    ]

    # Store them.
    split_data["train"].extend(train_records)
    split_data["test"].extend(test_records)
    split_data["validate"].extend(validate_records)

    print(
        f"\n{label.upper()} split:"
        f"\n  Train    : {len(train_records)}"
        f"\n  Test     : {len(test_records)}"
        f"\n  Validate : {len(validate_records)}"
    )


# =============================================================
# 9. COPY AUDIO FILES
# =============================================================

print("\nCopying audio files...")


# These counters help us see how many files were successfully
# copied and how many metadata entries had no matching audio.
copied_count = 0
missing_audio_count = 0


for split in ["train", "test", "validate"]:

    for record in split_data[split]:

        # UUID identifies the audio recording.
        uuid = str(record["uuid"])

        label = record["final_label"]

        # Find the original audio file using UUID.
        source_file = audio_files.get(uuid)

        # If the UUID is not found, we cannot copy the audio.
        if source_file is None:

            missing_audio_count += 1

            print(
                f"WARNING: Audio not found for UUID: {uuid}"
            )

            continue

        # Get the original filename.
        filename = os.path.basename(source_file)

        # Destination:
        #
        # output/train/dry/audio.wav
        # output/train/wet/audio.wav
        # etc.
        destination_folder = os.path.join(
            OUTPUT_FOLDER,
            split,
            label
        )

        destination_file = os.path.join(
            destination_folder,
            filename
        )

        # shutil.copy2() copies the file and also attempts to
        # preserve metadata such as modification time.
        shutil.copy2(
            source_file,
            destination_file
        )

        copied_count += 1


# =============================================================
# 10. FINAL SUMMARY
# =============================================================

print("\n" + "=" * 60)
print("DATASET CREATION COMPLETE")
print("=" * 60)

print(f"Total labeled samples : {len(labeled_df)}")
print(f"Audio files copied     : {copied_count}")
print(f"Missing audio files    : {missing_audio_count}")

print("\nFinal dataset:")

for split in ["train", "test", "validate"]:

    dry_folder = os.path.join(
        OUTPUT_FOLDER,
        split,
        "dry"
    )

    wet_folder = os.path.join(
        OUTPUT_FOLDER,
        split,
        "wet"
    )

    dry_count = len(os.listdir(dry_folder))
    wet_count = len(os.listdir(wet_folder))

    print(
        f"{split:10s} -> "
        f"Dry: {dry_count:5d} | "
        f"Wet: {wet_count:5d}"
    )

print("\nOutput folder:")
print(os.path.abspath(OUTPUT_FOLDER))