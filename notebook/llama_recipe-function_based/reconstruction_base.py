import os
import json
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage

from noise_pipeline import (
    SpectrogramModifier,
    NoisePipeline,
    ShapeFactory,
    PatternFactory,
    reconstruct_audio_from_final_spectrogram
)

# 재구성 함수들 (기존 코드 그대로 사용)
def reconstruct_pipeline_spectrogram_and_audio(parsed_explanation):
    """
    JSON 설명을 기반으로 스펙트로그램과 오디오를 재구성합니다.
    """
    spectrogram_base = parsed_explanation['spectrogram_base']
    shapes = parsed_explanation['shapes']
    patterns = parsed_explanation['patterns']

    sample_rate = spectrogram_base['sample_rate']
    n_fft = spectrogram_base['n_fft']
    hop_length = spectrogram_base['hop_length']
    noise_strength = spectrogram_base['noise_strength']
    noise_type = spectrogram_base['noise_type']
    noise_params = spectrogram_base['noise_params']

    # Initialize SpectrogramModifier
    spectro_mod = SpectrogramModifier(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        noise_strength=noise_strength,
        noise_type=noise_type,
        noise_params=noise_params
    )

    # Initialize NoisePipeline
    pipeline = NoisePipeline(
        spectro_mod=spectro_mod,
        apply_blur=False,
        blur_sigma=1.0
    )

    # Add shapes and patterns
    shape_factory = ShapeFactory()
    pattern_factory = PatternFactory()

    for shape_info in shapes:
        shape_name = shape_info['type']
        shape_params = shape_info['parameters']
        shape_obj = shape_factory.create(shape_name.lower(), **shape_params)
        pipeline.add_shape(shape_obj)

    for pattern_info in patterns:
        pattern_name = pattern_info['type']
        pattern_params = pattern_info['parameters']
        pattern_obj = pattern_factory.create(pattern_name.lower(), pattern_params)
        pipeline.add_pattern(pattern_obj)

    # Generate spectrogram and reconstruct audio
    duration = parsed_explanation.get('duration', 12.0)  # Default to 12.0 seconds
    signal_length = int(sample_rate * duration)
    silence_signal = np.zeros(signal_length)

    pipeline.generate(silence_signal)
    S_db = spectro_mod.S_db.copy()
    reconstructed_audio = reconstruct_audio_from_final_spectrogram(spectro_mod)

    return S_db, reconstructed_audio, sample_rate


def save_and_visualize_results(S_db, audio, sr, output_dir, file_id):
    """
    스펙트로그램과 오디오를 저장하고 시각화합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {file_id}")
    spectrogram_path = os.path.join(output_dir, f"{file_id}_spectrogram.png")
    plt.savefig(spectrogram_path, dpi=100)
    plt.close()

    # Save audio
    audio_path = os.path.join(output_dir, f"{file_id}.wav")
    sf.write(audio_path, audio, sr)

    print(f"Spectrogram saved at: {spectrogram_path}")
    print(f"Audio saved at: {audio_path}")

    return spectrogram_path, audio_path


def initialize_excel(headers, sheet_name="Compare Table"):
    """
    Excel 워크북을 초기화하고 헤더를 설정합니다.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(headers)
    return wb, ws


def adjust_cell_dimensions(ws, column, width, row=None, height=None):
    """
    Excel 셀의 크기를 조정합니다.
    """
    ws.column_dimensions[column].width = max(ws.column_dimensions[column].width or 0, width)
    if row and height:
        ws.row_dimensions[row].height = height


def insert_image_into_excel(ws, img_path, anchor):
    """
    지정된 위치에 이미지를 삽입합니다.
    """
    img = ExcelImage(img_path)
    img.anchor = anchor
    ws.add_image(img)


def process_sample(i, sample, ws, temp_dir, reconstructed_dir, output_dir):
    """
    단일 샘플을 처리하고 Excel에 데이터를 추가합니다.
    """
    # 텍스트 생성
    generated_text = inference_vllm(sample)

    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    print(f"recreated: {generated_text}")

    file_name = sample['file_name']
    complexity_level = sample['complexity_level']

    linear_spectrogram_with_axes_image = sample['spectrogram_with_axes'].convert("RGB")
    function_based_explanation_spectrogram = sample['function_based_explanation_spectrogram']

    # 원본 스펙트로그램 이미지 저장
    original_img_path = os.path.join(temp_dir, f"temp_original_{i}.png")
    plt.imshow(linear_spectrogram_with_axes_image)
    plt.axis('off')
    plt.savefig(original_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 이미지 크기 확인
    with PILImage.open(original_img_path) as img:
        img_width, img_height = img.size

    # 셀 크기 조정
    cell_width = img_width / 7  # 열 너비는 픽셀 크기의 1/7
    cell_height = img_height * 0.75  # 행 높이는 픽셀 크기의 0.75

    adjust_cell_dimensions(ws, 'E', cell_width, row=i + 2, height=cell_height)

    # JSON 설명을 파싱하여 재구성
    try:
        explanation_dict = json.loads(function_based_explanation_spectrogram)
        S_db, reconstructed_audio, sample_rate = reconstruct_pipeline_spectrogram_and_audio(explanation_dict)

        # 재구성된 스펙트로그램 이미지 저장
        reconstructed_file_id = f"reconstructed_{i}"
        reconstructed_spectrogram_path, _ = save_and_visualize_results(
            S_db, reconstructed_audio, sample_rate, reconstructed_dir, reconstructed_file_id
        )
    except json.JSONDecodeError:
        print(f"Sample {i}: JSON 디코딩 오류")
        reconstructed_spectrogram_path = None

    # 데이터 행 준비
    row = [
        file_name,
        complexity_level,
        function_based_explanation_spectrogram,
        generated_text
    ]

    # 원본 이미지 삽입
    insert_image_into_excel(ws, original_img_path, f"E{i + 2}")

    # 재구성된 이미지 삽입 (존재할 경우)
    if reconstructed_spectrogram_path and os.path.exists(reconstructed_spectrogram_path):
        insert_image_into_excel(ws, reconstructed_spectrogram_path, f"F{i + 2}")
    else:
        row.append("Reconstruction Failed")

    # 엑셀에 데이터 추가
    ws.append(row)


def clean_up_temp_files(temp_dirs, num_samples):
    """
    임시 이미지 파일을 삭제합니다.
    """
    for temp_dir in temp_dirs:
        for i in range(num_samples):
            # 원본 이미지 삭제
            original_img_path = os.path.join(temp_dir, f"temp_original_{i}.png")
            if os.path.exists(original_img_path):
                os.remove(original_img_path)

            # 재구성된 이미지 삭제
            reconstructed_img_path = os.path.join(temp_dir, f"reconstructed_{i}_spectrogram.png")
            if os.path.exists(reconstructed_img_path):
                os.remove(reconstructed_img_path)


def main(eval_dataset, inference_vllm, train_config, num_samples=100):
    """
    메인 실행 함수입니다.
    """
    # 엑셀 파일 설정
    output_excel = f"{train_config.output_dir}_compare_table.xlsx"
    headers = [
        "File Name", "Complexity Level",
        "Explanation About Spectrogram", "Generated Text",
        "Original Spectrogram Image", "Reconstructed Spectrogram Image"
    ]
    wb, ws = initialize_excel(headers)

    # 임시 이미지 저장 디렉토리 생성
    temp_dir = "temp_image"
    reconstructed_dir = "reconstructed_output"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)

    # 샘플 처리 루프
    for i in range(min(num_samples, len(eval_dataset))):
        sample = eval_dataset[i]
        process_sample(i, sample, ws, temp_dir, reconstructed_dir, "resconstructed_output")

    # 엑셀 저장
    wb.save(output_excel)
    print(f"Evaluation results saved to {output_excel}")

    # 임시 파일 정리
    clean_up_temp_files([temp_dir, reconstructed_dir], num_samples)
    print("Temporary files have been cleaned up.")


# 예시 사용법
if __name__ == "__main__":

    import os
    import torch
    from transformers import MllamaForConditionalGeneration, AutoTokenizer, AutoProcessor
    from llama_recipes.configs import train_config as TRAIN_CONFIG

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train_config = TRAIN_CONFIG()
    train_config.model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    train_config.num_epochs = 1
    train_config.run_validation = False
    train_config.gradient_accumulation_steps = 4
    train_config.batch_size_training = 1
    train_config.lr = 3e-4
    train_config.use_fast_kernels = True
    train_config.use_fp16 = True
    train_config.context_length = 512 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 1024 # T4 16GB or A10 24GB
    train_config.batching_strategy = "packing"
    train_config.output_dir = "ladlm_function_based_dataset_v2"
    train_config.use_peft = True

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = MllamaForConditionalGeneration.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
    processor = AutoProcessor.from_pretrained(train_config.model_name)

    import ladlm_function_based_dataset
    from torch.utils.data import DataLoader

    # Train/Test 데이터셋 로드
    train_dataset = ladlm_function_based_dataset.get_custom_dataset(
        dataset_config=None,
        processor=processor,
        split="train",
        split_ratio=0.9
    )

    eval_dataset = ladlm_function_based_dataset.get_custom_dataset(
        dataset_config=None,
        processor=processor,
        split="test",
        split_ratio=0.9
    )

    # Data Collator 생성
    data_collator = ladlm_function_based_dataset.get_data_collator(processor)

    # DataLoader 생성

    # do not use train
    # train_dataloader = DataLoader(
    #     train_dataset, 
    #     batch_size=train_config.batch_size_training, 
    #     collate_fn=data_collator, 
    #     shuffle=True
    # )

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=train_config.batch_size_training, 
        collate_fn=data_collator, 
        shuffle=False
    )

    print(f"Train DataLoader and Eval DataLoader ready!")

    eval_dataset = eval_dataset()
    inference_vllm = your_inference_function
    train_config = your_train_config_object
    main(eval_dataset, inference_vllm, train_config)

    pass  # 실제 사용 시 위 내용을 채워주세요.
