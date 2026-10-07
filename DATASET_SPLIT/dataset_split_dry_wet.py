import os
import shutil
import random
import pandas as pd


METADATA_FILE = "C:/Users/aydhi/OneDrive/Desktop/COUGH_CLASSIFIER_PROJECT/DATASET/DRY_WET/public_dataset_v3/metadata_compiled.csv"

SOURCE_AUDIO_FOLDER = "C:/Users/aydhi/OneDrive/Desktop/COUGH_CLASSIFIER_PROJECT/DATASET/DRY_WET/public_dataset_v3/coughvid_20211012"

OUTPUT_FOLDER = "C:/Users/aydhi/OneDrive/Desktop/COUGH_CLASSIFIER_PROJECT/DATASET/CLASSIFIED"

RANDOM_SEED = 42


COUGH_TYPE_COLUMNS = [
    "cough_type_1",
    "cough_type_2",
    "cough_type_3",
    "cough_type_4"
]

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".webm",
    ".m4a"
}


print("\nReading metadata...")

df = pd.read_csv(METADATA_FILE)

print(f"Total metadata rows: {len(df)}")


required_columns = ["uuid"] + COUGH_TYPE_COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"These required columns are missing from the CSV: "
        f"{missing_columns}"
    )


def get_cough_label(row):

    labels = []

    for column in COUGH_TYPE_COLUMNS:

        value = row[column]

        if pd.isna(value):
            return None

        value = str(value).strip().lower()

        labels.append(value)

    if all(label == "dry" for label in labels):
        return "dry"

    if all(label == "wet" for label in labels):
        return "wet"

    return None


df["final_label"] = df.apply(
    get_cough_label,
    axis=1
)


clean_df = df[
    df["final_label"].notna()
].copy()


print(
    f"\nClean DRY/WET recordings: {len(clean_df)}"
)


dry_count = (
    clean_df["final_label"] == "dry"
).sum()

wet_count = (
    clean_df["final_label"] == "wet"
).sum()

print("\nClean class distribution:")
print(f"DRY : {dry_count}")
print(f"WET : {wet_count}")


for split in ["train", "test", "validate"]:

    for label in ["dry", "wet"]:

        folder = os.path.join(
            OUTPUT_FOLDER,
            split,
            label
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


print("\nSearching for audio files...")

audio_files = {}


for root, directories, files in os.walk(
    SOURCE_AUDIO_FOLDER
):

    for filename in files:

        file_name_without_extension, extension = (
            os.path.splitext(filename)
        )

        extension = extension.lower()

        if extension not in AUDIO_EXTENSIONS:
            continue

        audio_files[
            file_name_without_extension
        ] = os.path.join(
            root,
            filename
        )


print(
    f"Audio files found: {len(audio_files)}"
)


random.seed(RANDOM_SEED)


split_data = {
    "train": [],
    "test": [],
    "validate": []
}


for label in ["dry", "wet"]:

    class_data = clean_df[
        clean_df["final_label"] == label
    ].copy()

    records = class_data.to_dict(
        "records"
    )

    random.shuffle(records)

    total = len(records)

    test_count = int(
        total * 0.05
    )

    validate_count = int(
        total * 0.05
    )

    train_count = (
        total
        - test_count
        - validate_count
    )

    train_records = records[
        :train_count
    ]

    test_records = records[
        train_count:
        train_count + test_count
    ]

    validate_records = records[
        train_count + test_count:
    ]

    split_data["train"].extend(
        train_records
    )

    split_data["test"].extend(
        test_records
    )

    split_data["validate"].extend(
        validate_records
    )

    print(
        f"\n{label.upper()} split:"
        f"\n  Train    : {len(train_records)}"
        f"\n  Test     : {len(test_records)}"
        f"\n  Validate : {len(validate_records)}"
    )


print("\nCopying audio files...")

copied_count = 0
missing_audio_count = 0


for split in ["train", "test", "validate"]:

    for record in split_data[split]:

        uuid = str(
            record["uuid"]
        )

        label = record[
            "final_label"
        ]

        source_file = audio_files.get(
            uuid
        )

        if source_file is None:

            missing_audio_count += 1

            print(
                f"WARNING: Audio not found: {uuid}"
            )

            continue

        filename = os.path.basename(
            source_file
        )

        destination_folder = os.path.join(
            OUTPUT_FOLDER,
            split,
            label
        )

        destination_file = os.path.join(
            destination_folder,
            filename
        )

        shutil.copy2(
            source_file,
            destination_file
        )

        copied_count += 1


print("\n" + "=" * 60)
print("DATASET CREATION COMPLETE")
print("=" * 60)

print(
    f"Clean DRY/WET metadata rows : "
    f"{len(clean_df)}"
)

print(
    f"Audio files copied          : "
    f"{copied_count}"
)

print(
    f"Missing audio files         : "
    f"{missing_audio_count}"
)


print("\nFinal dataset:")

for split in [
    "train",
    "test",
    "validate"
]:

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

    dry_files = len(
        os.listdir(dry_folder)
    )

    wet_files = len(
        os.listdir(wet_folder)
    )

    print(
        f"{split:10s} -> "
        f"Dry: {dry_files:5d} | "
        f"Wet: {wet_files:5d}"
    )


print("\nOutput folder:")
print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)