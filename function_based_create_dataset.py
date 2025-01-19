import os
import json
import pandas as pd
from datasets import Dataset, Features, Value, Audio, Image
import shutil
import sys


def round_floats(obj, precision=3):
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(elem, precision) for elem in obj]
    if isinstance(obj, float):
        return round(obj, precision)
    return obj


def process_json_files(output_json_dir, processed_json_dir):
    os.makedirs(processed_json_dir, exist_ok=True)

    json_files = [
        f for f in os.listdir(output_json_dir) if f.endswith('.json')
    ]
    if not json_files:
        print(f"No JSON files found in '{output_json_dir}'.")
        sys.exit(1)

    for filename in json_files:
        input_path = os.path.join(output_json_dir, filename)
        output_path = os.path.join(processed_json_dir, filename)

        with open(input_path, 'r') as f:
            data = json.load(f)

        data = round_floats(data, precision=3)

        function_based_explanation = {
            "spectrogram_base": data.pop("spectrogram_base"),
            "shapes": data.pop("shapes"),
            "patterns": data.pop("patterns"),
        }

        data['function_based_explanation_spectrogram'] = json.dumps(
            function_based_explanation, separators=(',', ':')
        )

        with open(output_path, 'w') as f:
            json.dump(data, f, separators=(',', ':'), ensure_ascii=False)


def create_dataset(processed_json_dir, output_dataset_dir):
    os.makedirs(output_dataset_dir, exist_ok=True)

    dataset = []

    for filename in os.listdir(processed_json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(processed_json_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                dataset.append(data)

    if not dataset:
        print(f"No data found in '{processed_json_dir}'.")
        sys.exit(1)

    df = pd.DataFrame(dataset)
    dataset_json_path = os.path.join(output_dataset_dir, 'dataset.json')
    df.to_json(dataset_json_path, orient='records', lines=True)


def load_dataset_from_json(dataset_json_path):
    df = pd.read_json(dataset_json_path, lines=True)

    if df.empty:
        print(f"The dataset at '{dataset_json_path}' is empty.")
        sys.exit(1)

    df = df.rename(columns={'file_path': 'audio'})

    features = Features({
        'audio': Audio(sampling_rate=16000),
        'file_name': Value('string'),
        'spectrogram_with_axes': Image(),
        'spectrogram_no_axes': Image(),
        'function_based_explanation_spectrogram': Value('string'),
        'complexity_level': Value('int32'),
        'shape_count': Value('int32'),
        'pattern_count': Value('int32'),
        'duration': Value('float32'),
    })

    return Dataset.from_pandas(df, features=features)


def save_dataset_as_arrow(dataset, output_dataset_dir):
    os.makedirs(output_dataset_dir, exist_ok=True)
    dataset.save_to_disk(output_dataset_dir)


def confirm_and_remove_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        if os.listdir(dir_path):
            while True:
                user_input = input(
                    f"The directory '{dir_path}' already exists and is not empty."
                    f" Delete? (y/n): "
                ).strip().lower()

                if user_input == 'y':
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path, exist_ok=True)
                    return True

                if user_input == 'n':
                    print("Operation cancelled.")
                    return False
    else:
        os.makedirs(dir_path, exist_ok=True)
    return True


def main():
    output_json_dir = "augmentation/output/json"
    processed_json_dir = "augmentation/processed_json"
    output_dataset_dir = "augmentation/dataset"

    if not os.path.exists(output_json_dir):
        print(f"The source directory '{output_json_dir}' does not exist.")
        sys.exit(1)

    if not any(
        f.endswith('.json') for f in os.listdir(output_json_dir)
    ):
        print(f"No valid JSON files found in '{output_json_dir}'.")
        sys.exit(1)

    directories = [processed_json_dir, output_dataset_dir]

    for dir_path in directories:
        if not confirm_and_remove_dir(dir_path):
            sys.exit(1)

    process_json_files(output_json_dir, processed_json_dir)
    create_dataset(processed_json_dir, output_dataset_dir)

    dataset_json_path = os.path.join(output_dataset_dir, 'dataset.json')
    dataset = load_dataset_from_json(dataset_json_path)
    dataset_size = len(dataset)

    dataset_name = f"function_based_{dataset_size}"
    hf_dataset_dir = f"augmentation/datasets/{dataset_name}"

    if not confirm_and_remove_dir(hf_dataset_dir):
        sys.exit(1)

    save_dataset_as_arrow(dataset, hf_dataset_dir)


if __name__ == "__main__":
    main()
