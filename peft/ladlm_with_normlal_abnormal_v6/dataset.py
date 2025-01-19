# dataset.py

import os
import copy
import json
import torch
from datasets import load_from_disk
import re


def get_custom_dataset(dataset_config, processor, dataset_path, split='train', split_ratio=0.9):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset at '{dataset_path}' does not exist.")

    dataset = load_from_disk(dataset_path)

    train_test_split = dataset.train_test_split(test_size=1 - split_ratio, shuffle=True, seed=42)
    return train_test_split[split]


def load_datasets(processor, dataset_path, split_ratio=0.9):
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


import re
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms

class CustomDataCollator:
    def __init__(self, processor, augmenter=None):
        self.processor = processor
        self.augmenter = augmenter
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, samples):
        dialogs, images = [], []

        for sample in samples:
            # 기본 데이터 추출
            file_name = sample['file_name']
            spectrogram_image = sample['linear_spectrogram_no_axes']['image'].convert("RGB")
            librosa_parameters = sample['linear_spectrogram_no_axes']['librosa_parameters']
            explanation = sample['explanation_about_spectrogram']
            machine_type = sample['machineType']
            label = sample['type']

            # 추가 데이터 추출
            audio_path = sample['file_path']
            float_features = sample['float_features']
            zero_crossing_rate = float_features['zero_crossing_rate']
            harmonic_to_noise_ratio = float_features['harmonic_to_noise_ratio']
            spectral_flatness = float_features['spectral_flatness']
            spectral_rolloff = float_features['spectral_rolloff']
            rms_energy = float_features['rms_energy']

            # 오디오 증강
            if self.augmenter:
                audio, sr = librosa.load(audio_path, sr=None)
                audio = self.augmenter.augment_audio(audio, sr)

            # 이미지 증강
            if self.augmenter:
                spectrogram_image = self.augmenter.augment_image(spectrogram_image)

            # 텍스트 설명 정리
            explanation = re.sub(r'\n+', '\n', explanation).strip()

            # 대화 형식 구성
            dialog = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                f"This is an image extracted from machine({machine_type}) noise using the Short-Time Fourier Transform (STFT) with librosa. "
                                f"The parameters used to extract this image are: {librosa_parameters}. "
                                f"Audio features include:\n"
                                f"- Zero Crossing Rate: {zero_crossing_rate}\n"
                                f"- Harmonic to Noise Ratio: {harmonic_to_noise_ratio}\n"
                                f"- Spectral Flatness: {spectral_flatness}\n"
                                f"- Spectral Rolloff: {spectral_rolloff}\n"
                                f"- RMS Energy: {rms_energy}\n\n"
                                f"Does this audio represent normal or anomaly?\n\n\n:\n"
                            )
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": 
                                f"explation: {explanation}\n"
                                f"normal or anomaly :{label}\n"
                        }
                    ]
                }
            ]

            # 대화 및 이미지 저장
            dialogs.append(dialog)
            images.append([spectrogram_image])  # Processor expects a list of image lists

        batch = tokenize_dialogs(dialogs, images, self.processor)
        return batch


def get_data_collator(processor):
    return CustomDataCollator(processor)
