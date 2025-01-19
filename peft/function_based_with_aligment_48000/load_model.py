# main.py

import os
import sys
import matplotlib.pyplot as plt
from config import load_train_config, load_lora_config, load_bnb_config, DATASET_PATH
from dataset import (
    load_datasets,
    save_eval_file_names,
    load_eval_file_names,
    get_full_eval_dataset,
    get_data_collator
)
from train import (
    load_model,
    prepare_peft_model,
    setup_optimizer_scheduler,
    train_model
)
from excel_table_generator import create_excel
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from inference import inference_vllm


def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    setup_environment()
    train_config = load_train_config()
    lora_config = load_lora_config()
    bnb_config = load_bnb_config()

    model, processor = load_model(train_config, bnb_config)
    train_dataset, eval_dataset = load_datasets(
        processor=processor,
        dataset_path=DATASET_PATH
    )

    try:
        model = PeftModel.from_pretrained(model, train_config.output_dir)
        print("Model loaded successfully from:", train_config.output_dir)
    except Exception as e:
        print(f"Failed to load the trained PEFT model: {e}")
        return

    print(model)

    sample = eval_dataset[0]

    generated_text = inference_vllm(model, processor, sample)

    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    if "<|eot_id|>" in generated_text:
        generated_text = generated_text.replace("<|eot_id|>", "").strip()

    print(f"Recreated: {generated_text}")

    file_name = sample['file_name']
    complexity_level = sample['complexity_level']
    spectrogram_image = sample['spectrogram_with_axes'].convert("RGB")

    # Save plot as image
    img_path = f"temp_image/temp_image.png"
    plt.imshow(spectrogram_image)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
