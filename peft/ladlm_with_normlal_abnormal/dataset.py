# dataset.py

import os
import copy
import json
import torch
from datasets import load_from_disk


def get_custom_dataset(dataset_config, processor, dataset_path, split='train', split_ratio=0.9):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset at '{dataset_path}' does not exist.")

    dataset = load_from_disk(dataset_path)

    train_test_split = dataset.train_test_split(test_size=1 - split_ratio, shuffle=True, seed=42)
    return train_test_split[split]


def load_datasets(processor, dataset_path, split_ratio=0.7):
    train_dataset = get_custom_dataset(
        dataset_config=None,
        processor=processor,
        dataset_path=dataset_path,
        split="train",
        split_ratio=split_ratio
    )
    eval_dataset = get_custom_dataset(
        dataset_config=None,
        processor=processor,
        dataset_path=dataset_path,
        split="test",
        split_ratio=split_ratio
    )
    return train_dataset, eval_dataset


def get_full_eval_dataset(dataset, eval_file_names):
    return [sample for sample in dataset if sample['file_name'] in eval_file_names]


def save_eval_file_names(eval_dataset, save_path):
    file_names = [sample['file_name'] for sample in eval_dataset]
    with open(save_path, "w") as f:
        json.dump(file_names, f, indent=4)


def load_eval_file_names(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def check_header(targets, seq):
    for i in range(len(seq) - 2):
        if seq[i:i + 3] in targets:
            return True
    return False


def replace_target(target, seq):
    for i in range(len(seq) - 2):
        if seq[i:i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(dialogs, images, processor):
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
        prompt_header_seqs = [
            [128006, 9125, 128007],
            [128006, 882, 128007]
        ]
        for idx in eot_indices:
            current_seq = labels[last_idx:idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                labels[last_idx:idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        labels = [
            -100 if (token == processor.tokenizer.pad_token_id or token == 128256)
            else token
            for token in labels
        ]
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"  # Ensure padding on the right

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            file_name = sample['file_name']

            # Extract and convert images
            # linear_spectrogram_with_axes_image = sample['linear_spectrogram_with_axes']['image'].convert("RGB") # we can use this?
            linear_spectrogram_no_axes_image = sample['linear_spectrogram_no_axes']['image'].convert("RGB")
            
            # Optionally, choose which spectrogram to use
            image = linear_spectrogram_no_axes_image

            # Extract parameters
            librosa_parameters = sample['linear_spectrogram_no_axes']['librosa_parameters']
            
            # Extract other features
            
            explanation_about_spectrogram = sample['explanation_about_spectrogram']
            machine_type = sample['machineType']
            label = sample['type']

            # Replace multiple newlines with a single newline
            # explanation_about_spectrogram = re.sub(r'\n+', '\n', explanation_about_spectrogram)
            dialog = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": f"This is an image extracted from machine({machine_type}) noise using the Short-Time Fourier Transform (STFT) with librosa."
                                 f"The parameters used to extract this image are as follows: {librosa_parameters}. "
                                 f"\nExplanation about spectrogram and Detection anomaly:\n"}
                ]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", 
                         "text": f"explanation about spetrogram: {explanation_about_spectrogram}\n"
                                 f"label :{label}\n" }
                    ]
                }
            ]
            # print(dialog)
            dialogs.append(dialog)  # Each sample has a list of dialog lists
            images.append([image])     # Processor expects a list of image lists

        batch = tokenize_dialogs(dialogs, images, self.processor)
        return batch

def get_data_collator(processor):
    return CustomDataCollator(processor)
