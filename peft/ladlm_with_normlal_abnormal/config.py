# config.py

from dataclasses import asdict
from llama_recipes.configs import train_config as TRAIN_CONFIG, lora_config as LORA_CONFIG
import torch
from transformers import BitsAndBytesConfig


DATASET_PATH = '/home/uturtle/repos/ladlm/datasets/ladlm'


def load_train_config():
    config = TRAIN_CONFIG()
    config.model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    config.num_epochs = 1
    config.run_validation = False
    config.gradient_accumulation_steps = 4
    config.batch_size_training = 1
    config.lr = 3e-4
    config.use_fast_kernels = True
    config.use_fp16 = True
    total_memory = torch.cuda.get_device_properties(0).total_memory
    config.context_length = 512 if total_memory < 16e9 else 1024
    config.batching_strategy = "packing"
    config.output_dir = "ladlm_with_normlal_abnormal"
    config.use_peft = True
    config.weight_decay = 0.01
    config.gamma = 0.1
    return config


def load_lora_config():
    config = LORA_CONFIG()
    config.r = 8
    config.lora_alpha = 32
    config.lora_dropout = 0.02
    return config


def load_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
