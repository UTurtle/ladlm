import os
import sys
import logging
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

# 로깅 설정
logging.basicConfig(
    filename="training_log.txt",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logger.info("Environment setup completed.")

def main():
    try:
        setup_environment()
        logger.info("Starting main process...")

        # Config 로드
        train_config = load_train_config()
        lora_config = load_lora_config()
        bnb_config = load_bnb_config()
        logger.info("Configuration files loaded successfully.")

        # 모델 및 데이터 로드
        model, processor = load_model(train_config, bnb_config)
        train_dataset, eval_dataset = load_datasets(
            processor=processor,
            dataset_path=DATASET_PATH
        )
        logger.info(f"Train dataset count: {len(train_dataset)}")
        logger.info(f"Eval dataset count: {len(eval_dataset)}")

        # DataLoader 준비
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
        logger.info("Train DataLoader and Eval DataLoader ready!")

        # Training 시작
        model = prepare_peft_model(model, lora_config)
        optimizer, scheduler = setup_optimizer_scheduler(model, train_config)

        logger.info("Training started...")
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
        logger.info("Model saved to output directory.")

        create_excel(train_config, eval_dataset, processor, model)
        logger.info("Excel table created with evaluation results.")

    except Exception as e:
        logger.error("An error occurred during execution.", exc_info=True)

if __name__ == "__main__":
    main()
