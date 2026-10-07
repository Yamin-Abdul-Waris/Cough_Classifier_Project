"""
===============================================================
WHOOPING / PERTUSSIS COUGH DATASET SPLITTER
===============================================================

WHAT WE ARE BUILDING
--------------------

Metadata
   |
   v
Take only rows where:
    type = "p"
   |
   v
These are WHOOPING / PERTUSSIS coughs
   |
   v
Randomly shuffle them
   |
   v
Split:
    95% -> train
     5% -> validate
   |
   v
Copy audio files into:

OUTPUT/
├── train/
│   └── whooping/
│
└── validate/
    └── whooping/

IMPORTANT:
----------
1. Test data is NOT created or modified here.
   You said your test set is already done.

2. Only type = "p" is selected.

3. The split is RANDOM, not based on the original
   order of the metadata.

4. RANDOM_SEED makes the split reproducible.

5. The original audio files are COPIED, not moved/deleted.

6. This code assumes the audio filename matches sound_id:

       sound_id = a1
       audio file = a1.wav

   If your files have another naming format, change the
   matching section accordingly.

===============================================================
"""

import os
import shutil
import random
import pandas as pd


# =============================================================
# 1. CONFIGURATION
# =============================================================

# Metadata CSV file.
METADATA_FILE = (
    "C:/Users/aydhi/OneDrive/Desktop/"
    "COUGH_CLASSIFIER_PROJECT/DATASET/METADATA/"
    "metadata_whooping.csv"
)

# Folder containing the original whooping cough audio files.
SOURCE_AUDIO_FOLDER = (
    "C:/Users/aydhi/OneDrive/Desktop/"
    "COUGH_CLASSIFIER_PROJECT/DATASET/WHOOPING COUGH/WHOOPING COUGH_1/Sound Data"
)

# Folder where train/validate will be created.
OUTPUT_FOLDER = (
    "C:/Users/aydhi/OneDrive/Desktop/"
    "COUGH_CLASSIFIER_PROJECT/DATASET/"
    "CLASSIFIED"
)

# Fixed seed so the random split can be reproduced.
RANDOM_SEED = 42


# =============================================================
# 2. READ METADATA
# =============================================================

print("\nReading metadata...")

# pandas reads the CSV into a DataFrame (table).
df = pd.read_csv(METADATA_FILE)

print(f"Total metadata entries: {len(df)}")


# =============================================================
# 3. CHECK REQUIRED COLUMNS
# =============================================================

# We need:
#   sound_id -> identifies the audio file
#   type     -> tells us the cough type
required_columns = [
    "sound_id",
    "type"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    # Stop the program if the metadata format is incorrect.
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =============================================================
# 4. SELECT ONLY WHOOPING / PERTUSSIS COUGHS
# =============================================================

# Convert type to lowercase and remove accidental spaces.
#
# We then keep ONLY:
#
#     type == "p"
#
# Everything else is ignored.
whooping_df = df[
    df["type"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "p"
].copy()


print(
    f"Whooping / pertussis recordings found: "
    f"{len(whooping_df)}"
)


# =============================================================
# 5. RANDOM SHUFFLE
# =============================================================

# Set the random seed.
random.seed(RANDOM_SEED)

# Convert the DataFrame rows into a Python list.
records = whooping_df.to_dict(
    "records"
)

# Shuffle the recordings randomly.
#
# This prevents the original metadata order from deciding
# which recordings go into train/validate.
random.shuffle(records)


# =============================================================
# 6. 95% TRAIN / 5% VALIDATE
# =============================================================

total = len(records)

# Calculate 5% for validation.
validate_count = round(
    total * 0.05
)

# The remaining recordings go to training.
train_count = (
    total - validate_count
)

# First 95% -> train
train_records = records[
    :train_count
]

# Remaining 5% -> validate
validate_records = records[
    train_count:
]


print("\nSplit:")
print(
    f"Train    : {len(train_records)}"
)
print(
    f"Validate : {len(validate_records)}"
)


# =============================================================
# 7. CREATE OUTPUT FOLDERS
# =============================================================

train_folder = os.path.join(
    OUTPUT_FOLDER,
    "train",
    "whooping"
)

validate_folder = os.path.join(
    OUTPUT_FOLDER,
    "validate",
    "whooping"
)

# Create folders if they don't already exist.
os.makedirs(
    train_folder,
    exist_ok=True
)

os.makedirs(
    validate_folder,
    exist_ok=True
)


# =============================================================
# 8. FIND AUDIO FILES
# =============================================================

print("\nSearching for audio files...")

# Dictionary:
#
#     sound_id -> audio file path
#
# Example:
#
#     a1 -> C:/.../a1.wav
#
audio_files = {}

# Search through the source folder and all subfolders.
for root, directories, files in os.walk(
    SOURCE_AUDIO_FOLDER
):

    for filename in files:

        # Separate filename and extension.
        file_name, extension = os.path.splitext(
            filename
        )

        # Store the file using its filename without extension.
        #
        # a1.wav -> key = "a1"
        audio_files[file_name] = os.path.join(
            root,
            filename
        )


print(
    f"Audio files found: {len(audio_files)}"
)


# =============================================================
# 9. COPY TRAIN AUDIO
# =============================================================

print("\nCopying training files...")

train_copied = 0
train_missing = 0

for record in train_records:

    # Get the sound ID from metadata.
    sound_id = str(
        record["sound_id"]
    ).strip()

    # Find corresponding audio.
    source_file = audio_files.get(
        sound_id
    )

    if source_file is None:

        print(
            f"WARNING: Audio not found: {sound_id}"
        )

        train_missing += 1
        continue

    # Keep the original filename.
    filename = os.path.basename(
        source_file
    )

    destination = os.path.join(
        train_folder,
        filename
    )

    # Copy instead of moving.
    shutil.copy2(
        source_file,
        destination
    )

    train_copied += 1


# =============================================================
# 10. COPY VALIDATION AUDIO
# =============================================================

print("\nCopying validation files...")

validate_copied = 0
validate_missing = 0

for record in validate_records:

    sound_id = str(
        record["sound_id"]
    ).strip()

    source_file = audio_files.get(
        sound_id
    )

    if source_file is None:

        print(
            f"WARNING: Audio not found: {sound_id}"
        )

        validate_missing += 1
        continue

    filename = os.path.basename(
        source_file
    )

    destination = os.path.join(
        validate_folder,
        filename
    )

    shutil.copy2(
        source_file,
        destination
    )

    validate_copied += 1


# =============================================================
# 11. FINAL SUMMARY
# =============================================================

print("\n" + "=" * 60)
print("WHOOPING DATASET CREATION COMPLETE")
print("=" * 60)

print(
    f"Total 'p' recordings : {total}"
)

print(
    f"Train                : {len(train_records)}"
)

print(
    f"Validate             : {len(validate_records)}"
)

print(
    f"Train files copied   : {train_copied}"
)

print(
    f"Validate files copied: {validate_copied}"
)

print(
    f"Train missing        : {train_missing}"
)

print(
    f"Validate missing     : {validate_missing}"
)

print("\nOutput folder:")
print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)