import copy
import itertools
from datasets import load_from_disk, Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# Helper Functions
def check_header(targets, seq):
    """
    Check if any 3-token sequence in `seq` matches any target in `targets`.
    """
    for i in range(len(seq) - 2):
        if seq[i:i + 3] in targets:
            return True
    return False

def replace_target(target, seq):
    """
    Replace occurrences of `target` in `seq` with -100.
    """
    for i in range(len(seq) - 2):
        if seq[i:i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq

def tokenize_dialogs(dialogs, images, processor):
    """
    Tokenize the dialogs and images using the provided processor.
    Masks certain tokens as specified.
    """
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(
        images=images,
        text=text_prompt,
        padding=True,
        return_tensors="pt"
    )
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [idx for idx, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # Define prompt header sequences
        prompt_header_seqs = [
            [128006, 9125, 128007],  # <|start_header_id|>system<|end_header_id|>
            [128006, 882, 128007]    # <|start_header_id|>user<|end_header_id|>
        ]
        for idx in eot_indices:
            current_seq = labels[last_idx:idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # Mask the prompt header sequence
                labels[last_idx:idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
        # Mask assistant header prompt <|start_header_id|>assistant<|end_header_id|>
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask padding tokens and image tokens (128256)
        labels = [
            -100 if (token == processor.tokenizer.pad_token_id or token == 128256) else token
            for token in labels
        ]
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch

# Custom Dataset Loading Function
def get_custom_dataset(dataset_config, processor, dataset_path, split='train', split_ratio=0.9):
    """
    Load and preprocess the custom dataset.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset at '{dataset_path}' does not exist.")

    dataset = load_from_disk(dataset_path)
    # dataset = dataset_dict['train']

    # Optionally select a subset for quick testing
    # Remove or adjust the following line to use the full dataset
    # dataset = dataset.select(range(200))  # Example: select first 200 samples

    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=1 - split_ratio, shuffle=True, seed=42)
    return train_test_split[split]

# Custom Data Collator
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"  # Ensure padding on the right

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            file_name = sample['file_name']

            # Extract and convert images
            # spectrogram_with_axes = sample['spectrogram_with_axes'].convert("RGB") # we can use this?
            spectrogram_no_axes = sample['spectrogram_no_axes'].convert("RGB")
            
            # Optionally, choose which spectrogram to use
            image = spectrogram_no_axes

            # Extract other features
            function_based_explanation_spectrogram = sample['function_based_explanation_spectrogram']

            # Replace multiple newlines with a single newline
            # function_based_explanation_spectrogram = re.sub(r'\n+', '\n', function_based_explanation_spectrogram)

            # Create dialogs as per your specification
            dialog = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": f"This is an image extracted using the Short-Time Fourier Transform (STFT) with librosa. "
                                 f"\spectrogram json:\n"}
                ]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": function_based_explanation_spectrogram}
                    ]
                }
            ]
            dialogs.append(dialog)  # Each sample has a list of dialog lists
            images.append([image])     # Processor expects a list of image lists

        # Tokenize the dialogs and images
        batch = tokenize_dialogs(dialogs, images, self.processor)
        return batch

# Function to Get Data Collator
def get_data_collator(processor):
    return CustomDataCollator(processor)

