import os
import shutil
import random
import pandas as pd


METADATA_FILE = (
    "C:/Users/aydhi/OneDrive/Desktop/"
    "COUGH_CLASSIFIER_PROJECT/DATASET/METADATA/"
    "metadata_whooping.csv"
)

SOURCE_AUDIO_FOLDER = (
    "C:/Users/aydhi/OneDrive/Desktop/"
    "COUGH_CLASSIFIER_PROJECT/DATASET/"
    "WHOOPING COUGH/WHOOPING COUGH_1/Sound Data"
)

OUTPUT_FOLDER = (
    "C:/Users/aydhi/OneDrive/Desktop/"
    "COUGH_CLASSIFIER_PROJECT/DATASET/"
    "CLASSIFIED"
)

RANDOM_SEED = 42


print("\nReading metadata...")

df = pd.read_csv(METADATA_FILE)

print(
    f"Total metadata entries: {len(df)}"
)


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

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


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


random.seed(RANDOM_SEED)


records = whooping_df.to_dict(
    "records"
)

random.shuffle(records)


total = len(records)

validate_count = round(
    total * 0.05
)

train_count = (
    total - validate_count
)


train_records = records[
    :train_count
]

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


os.makedirs(
    train_folder,
    exist_ok=True
)

os.makedirs(
    validate_folder,
    exist_ok=True
)


print("\nSearching for audio files...")


audio_files = {}


for root, directories, files in os.walk(
    SOURCE_AUDIO_FOLDER
):

    for filename in files:

        file_name, extension = os.path.splitext(
            filename
        )

        audio_files[file_name] = os.path.join(
            root,
            filename
        )


print(
    f"Audio files found: {len(audio_files)}"
)


print("\nCopying training files...")


train_copied = 0
train_missing = 0


for record in train_records:

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

        train_missing += 1
        continue

    filename = os.path.basename(
        source_file
    )

    destination = os.path.join(
        train_folder,
        filename
    )

    shutil.copy2(
        source_file,
        destination
    )

    train_copied += 1


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