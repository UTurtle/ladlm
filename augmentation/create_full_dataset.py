# create_full_dataset.py

import os
import json
import pandas as pd
from datasets import Dataset, Features, Value, Audio, Image

def round_floats(obj, precision=3):
    """
    Recursively rounds float values in a JSON-like object to the specified precision.
    """
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(elem, precision) for elem in obj]
    elif isinstance(obj, float):
        return round(obj, precision)
    else:
        return obj

def process_json_files(output_json_dir, processed_json_dir):
    """
    Processes JSON files by removing redundant level information, rounding floats,
    and converting 'function_based_explanation_spectrogram' to a JSON string.
    
    Parameters:
    - output_json_dir (str): Path to the original JSON files.
    - processed_json_dir (str): Path to save the processed JSON files.
    """
    os.makedirs(processed_json_dir, exist_ok=True)
    
    for filename in os.listdir(output_json_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(output_json_dir, filename)
            output_path = os.path.join(processed_json_dir, filename)
            
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # 파일 이름 형식 변경 (중복 레벨 제거)
            for key in ['file_path', 'file_name', 'spectrogram_with_axes', 'spectrogram_no_axes']:
                original_value = data.get(key, "")
                parts = original_value.split('_level_')
                if len(parts) > 2:
                    # 첫 번째 '_level_' 이후 중복 제거
                    new_value = '_level_'.join([parts[0], parts[1]] + parts[2:])
                    # 중복된 'level_{level}' 제거
                    new_value = new_value.replace(f'_level_{data["complexity_level"]}_level_{data["complexity_level"]}', f'_level_{data["complexity_level"]}')
                    data[key] = new_value
            
            # 소수점 반올림
            data = round_floats(data, precision=3)
            
            # 'function_based_explanation_spectrogram' 필드를 문자열로 변환
            function_based_explanation = {
                "spectrogram_base": data.pop("spectrogram_base"),
                "shapes": data.pop("shapes"),
                "patterns": data.pop("patterns")
            }
            # JSON 문자열로 변환하면서 공백 제거
            function_based_explanation_str = json.dumps(function_based_explanation, separators=(',', ':'))
            data['function_based_explanation_spectrogram'] = function_based_explanation_str
            
            # JSON 파일 저장
            with open(output_path, 'w') as f:
                json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
    
    print(f"Processed JSON files are saved in: {processed_json_dir}")

def create_dataset(processed_json_dir, output_dataset_dir):
    """
    Creates a dataset by aggregating processed JSON files.
    
    Parameters:
    - processed_json_dir (str): Directory containing processed JSON files.
    - output_dataset_dir (str): Directory to save the aggregated dataset.
    """
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    dataset = []
    
    for filename in os.listdir(processed_json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(processed_json_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                dataset.append(data)
    
    # DataFrame으로 변환
    df = pd.DataFrame(dataset)
    
    # 데이터셋을 JSON Lines 형식으로 저장
    dataset_json_path = os.path.join(output_dataset_dir, 'dataset.json')
    df.to_json(dataset_json_path, orient='records', lines=True)
    
    print(f"Aggregated dataset saved to: {dataset_json_path}")

def load_dataset_from_json(dataset_json_path):
    """
    Loads the dataset from a JSON Lines file into a Hugging Face Dataset.
    
    Parameters:
    - dataset_json_path (str): Path to the dataset.json file.
    
    Returns:
    - Dataset: Hugging Face Dataset object.
    """
    # pandas를 사용하여 데이터 로드
    df = pd.read_json(dataset_json_path, lines=True)
    
    # 'file_path' 컬럼을 'audio'로 이름 변경
    df = df.rename(columns={'file_path': 'audio'})
    
    # Features 정의
    features = Features({
        'audio': Audio(sampling_rate=16000),  
        'file_name': Value('string'),
        'spectrogram_with_axes': Image(),
        'spectrogram_no_axes': Image(),
        'function_based_explanation_spectrogram': Value('string'),  # JSON 문자열
        'complexity_level': Value('int32'),
        'shape_count': Value('int32'),
        'pattern_count': Value('int32'),
        'duration': Value('float32')
    })
    
    # Hugging Face Dataset 생성
    dataset = Dataset.from_pandas(df, features=features)
    
    return dataset

def save_dataset_as_arrow(dataset, output_dataset_dir):
    """
    Saves the Hugging Face Dataset in Arrow format.
    
    Parameters:
    - dataset (Dataset): Hugging Face Dataset object.
    - output_dataset_dir (str): Directory to save the Arrow files.
    """
    os.makedirs(output_dataset_dir, exist_ok=True)
    dataset.save_to_disk(output_dataset_dir)
    print(f"Dataset saved to {output_dataset_dir}")

def main():
    # 직접 변수에 값 할당
    output_json_dir = "output/json"
    processed_json_dir = "processed_json"
    output_dataset_dir = "dataset"
    hf_dataset_dir = "datasets/ladlm_function_based_dataset"
    complexity_level = 5  # 필요에 따라 변경

    process_json_files(output_json_dir, processed_json_dir)
    create_dataset(processed_json_dir, output_dataset_dir)
    dataset = load_dataset_from_json(os.path.join(output_dataset_dir, 'dataset.json'))
    save_dataset_as_arrow(dataset, hf_dataset_dir)

if __name__ == "__main__":
    main()
