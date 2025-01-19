# excel_table_generator.py

import os
import json
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import torch
from inference import inference_vllm


def create_excel(train_config, eval_dataset, processor, model):
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

    # Create temporary image directory
    if not os.path.exists("temp_image"):
        os.makedirs("temp_image")

    # Process evaluation dataset
    max_len = min(100, len(eval_dataset))
    for i in range(max_len):
        sample = eval_dataset[i]
        
        generated_text = inference_vllm(model, processor, sample)

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
        ws.column_dimensions['E'].width = max(ws.column_dimensions['E'].width or 0, img_width / 7)
        ws.row_dimensions[i + 2].height = img_height * 0.75

        # Append data
        ws.append([
            file_name,
            complexity_level,
            sample['function_based_explanation_spectrogram'],
            generated_text
        ])

        # Insert image
        img = ExcelImage(img_path)
        img.anchor = f"E{i + 2}"
        ws.add_image(img)

    # Save Excel file
    wb.save(output_excel)

    # Remove temporary images
    for i in range(max_len):
        img_path = f"temp_image/temp_image_{i}.png"
        if os.path.exists(img_path):
            os.remove(img_path)

    print(f"Evaluation results saved to {output_excel}")
