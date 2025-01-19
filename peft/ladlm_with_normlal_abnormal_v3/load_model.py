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

    base_model, processor = load_model(train_config, bnb_config)
    train_dataset, eval_dataset = load_datasets(
        processor=processor,
        dataset_path=DATASET_PATH
    )

    lora_model = PeftModel.from_pretrained(base_model, train_config.output_dir)
    print("Model loaded successfully from:", train_config.output_dir)


    sample = eval_dataset[0]
    
    generated_text = inference_vllm(base_model, processor, sample)

    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    print(f"Recreated: {generated_text}")

    create_excel(train_config, eval_dataset, processor, base_model)

if __name__ == "__main__":
    main()
