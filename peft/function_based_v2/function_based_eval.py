import os
import json
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
import matplotlib.pyplot as plt
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from llama_recipes.configs import train_config as TRAIN_CONFIG
from peft import PeftModel
from function_based_dataset import get_custom_dataset, get_data_collator
from torch.utils.data import DataLoader


# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Train configuration
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
train_config.output_dir = "function_based_v2"
train_config.use_peft = True

dataset_path = '../../datasets/function_based_10000'

# Model configuration
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
    torch_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(train_config.model_name)

# Load evaluation file names
def load_eval_file_names(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

eval_file_names = load_eval_file_names("eval_file_names.json")

dataset = get_custom_dataset(
    dataset_config=None,
    processor=processor,
    dataset_path=dataset_path,
)

def get_eval_dataset(dataset, eval_file_names):
    return [sample for sample in dataset if sample['file_name'] in eval_file_names]


eval_dataset = get_eval_dataset(dataset, eval_file_names)

# Load LoRA model
lora_model = PeftModel.from_pretrained(model, train_config.output_dir)

# Inference function
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
                     "text": "This is an image extracted using the Short-Time Fourier Transform (STFT) with librosa.\n"}
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

    return processor.decode(output[0])

# Create Excel file
output_excel = f"{train_config.output_dir}_compare_table.xlsx"
wb = Workbook()
ws = wb.active
ws.title = "Compare Table"

# Add headers
headers = [
    "File Name", "Complexity Level",
    "Explanation About Spectrogram", "Generated Text", "Spectrogram Image"
]
ws.append(headers)

if not os.path.exists("temp_image"):
    os.makedirs("temp_image")

# Process evaluation dataset
for i, sample in enumerate(eval_dataset[:100]):
    generated_text = inference_vllm(sample)

    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    print(f"Recreated: {generated_text}")

    file_name = sample['file_name']
    complexity_level = sample['complexity_level']
    spectrogram_image = sample['spectrogram_with_axes'].convert("RGB")

    # Save plot as image
    img_path = f"temp_image/temp_image_{i}.png"
    plt.imshow(spectrogram_image)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Get image dimensions
    with PILImage.open(img_path) as img:
        img_width, img_height = img.size

    # Adjust cell dimensions
    ws.column_dimensions['G'].width = max(ws.column_dimensions['G'].width or 0, img_width / 7)
    ws.row_dimensions[i + 2].height = img_height * 0.75

    # Append data
    ws.append([file_name, complexity_level, sample['function_based_explanation_spectrogram'], generated_text])

    # Insert image
    img = ExcelImage(img_path)
    img.anchor = f"G{i + 2}"
    ws.add_image(img)

# Save Excel file
wb.save(output_excel)

# Remove temporary images
for i in range(len(eval_dataset)):
    img_path = f"temp_image/temp_image_{i}.png"
    if os.path.exists(img_path):
        os.remove(img_path)

print(f"Evaluation results saved to {output_excel}")
