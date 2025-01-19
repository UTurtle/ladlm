# train.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers import MllamaForConditionalGeneration, AutoProcessor
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from llama_recipes.utils.train_utils import train
from config import load_lora_config
from dataclasses import asdict


def load_model(train_config, bnb_config):
    model = MllamaForConditionalGeneration.from_pretrained(
        train_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(train_config.model_name)
    return model, processor


def prepare_peft_model(model, lora_config):
    peft_config = LoraConfig(**asdict(lora_config))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model


def setup_optimizer_scheduler(model, train_config):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    return optimizer, scheduler


def train_model(model, train_dataloader, eval_dataloader, processor,
               optimizer, scheduler, train_config):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        processor,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run=None,
    )
    return results
