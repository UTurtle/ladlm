import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import MllamaForConditionalGeneration, AutoTokenizer, AutoProcessor
from llama_recipes.configs import train_config as TRAIN_CONFIG
from transformers import BitsAndBytesConfig
from function_based_dataset import get_custom_dataset, get_data_collator
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG
from llama_recipes.utils.train_utils import train
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json

# Configurations
train_config = TRAIN_CONFIG()
train_config.model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
train_config.num_epochs = 1
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 512 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 1024
train_config.batching_strategy = "packing"
train_config.output_dir = "function_based_v1"
train_config.use_peft = True

dataset_path = '../../datasets/function_based_100'

# Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Model and Processor
model = MllamaForConditionalGeneration.from_pretrained(
    train_config.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(train_config.model_name)

# Dataset Loading
train_dataset = get_custom_dataset(
    dataset_config=None,
    processor=processor,
    dataset_path=dataset_path,
    split="train",
    split_ratio=0.7
)

eval_dataset = get_custom_dataset(
    dataset_config=None,
    processor=processor,
    dataset_path=dataset_path,
    split="test",
    split_ratio=0.7
)

print(f"훈련 데이터셋의 개수: {len(train_dataset)}")
print(f"평가 데이터셋의 개수: {len(eval_dataset)}")

# Data Collator and DataLoader
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

print(f"Train DataLoader and Eval DataLoader ready!")

# Save Evaluation File Names
def save_eval_file_names(eval_dataset, save_path):
    file_names = [sample['file_name'] for sample in eval_dataset]
    with open(save_path, "w") as f:
        json.dump(file_names, f, indent=4)

file_name_save_path = "eval_file_names.json"
save_eval_file_names(eval_dataset, file_name_save_path)

# Inference Function
def inference_vllm(sample, max_token=512):
    file_name = sample['file_name']

    spectrogram_with_axes = sample['spectrogram_with_axes'].convert("RGB")
    spectrogram_no_axes = sample['spectrogram_no_axes'].convert("RGB")

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "This is an image extracted using the Short-Time Fourier Transform (STFT) with librosa. "
                             "\nspectrogram json:\n"}
                ],
            },
        ]
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        spectrogram_no_axes,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_token)

    print(processor.decode(output[0]))
    plt.imshow(spectrogram_with_axes)
    plt.axis('off')
    plt.show()

    return processor.decode(output[0])

# LoRA Configuration
lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_config.lora_dropout = 0.02

peft_config = LoraConfig(**asdict(lora_config))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Training Setup
model.train()
optimizer = optim.AdamW(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=train_config.weight_decay,
)
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Training
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

# Save Model
model.save_pretrained(train_config.output_dir)
