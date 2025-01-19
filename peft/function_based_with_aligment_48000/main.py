# main.py

import os
import sys
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
    print(f"Train dataset count: {len(train_dataset)}")
    print(f"Eval dataset count: {len(eval_dataset)}")

    data_collator = get_data_collator(processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size_training,
        collate_fn=data_collator,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=train_config.batch_size_training,
        collate_fn=data_collator,
        shuffle=False
    )
    print("Train DataLoader and Eval DataLoader ready!")

    # train
    model = prepare_peft_model(model, lora_config)
    optimizer, scheduler = setup_optimizer_scheduler(model, train_config)

    model.config.use_cache=False
    results = train_model(
        model,
        train_dataloader,
        eval_dataloader,
        processor,
        optimizer,
        scheduler,
        train_config
    )

    model.save_pretrained(train_config.output_dir)
    create_excel(train_config, eval_dataset, processor, model)


if __name__ == "__main__":
    main()
