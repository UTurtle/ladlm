
# Datasest Input

### Input dataset discussion result

- input?
    - no axis stft features: 2d
    - librosa feature parameters: str
    - prompt (Explanation about spectrogram): str
- output?
    - Explanation about spectrogram: str
    - Spectrogram w/ axis: 2d (skip)

### How to set data format?

should make two program
- `ladlm_extract_json_to_dataset.py` : for json to dataset
- `ladlm_dataset.py` : for dataset fit to mllama model (refer this: `ocrvqa_dataset.py`)

- Json
    - extracting
        - no axis stft features: 2d             -> input
        - librosa feature parameters: str       -> input
        - other infomations (if need)           -> input (skip)
        - Explanation about spectrogram: str    -> output

to

```python
image = sample['images'][0].convert("RGB")
dialog = sample['texts']

dialogs = [
    [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", 
                "text": f"This is an image extracted using the Short-Time Fourier Transform (STFT) with librosa. The parameters used to extract this image are as follows: {dialog["librosa_feature_parameter"]}.\n {dialog["user_prompt"]}:"
            }
        ]},
        {"role" : "assistant" , "content":[
            {"type": "text", "text": dialog["explanation_about_spectrogram"]} 
        ]}
    ]
]
```

### In Training

```python
### Training data
image = sample['images'][0].convert("RGB")
dialog = sample['texts']

dialogs = [
    [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", 
                "text": f"This is an image extracted using the Short-Time Fourier Transform (STFT) with librosa. The parameters used to extract this image are as follows: {dialog["librosa_feature_parameter"]}.\n {dialog["user_prompt"]}:"
            }
        ]},
        {"role" : "assistant" , "content":[
            {"type": "text", "text": dialog["explanation_about_spectrogram"]} 
        ]}
    ]
]
# preprocessed = tokenize_dialogs(dialogs,images, self.processor) # not real. 
```

- `tokenize_dialogs(dialogs,images, self.processor)`
    - `text_prompt = processor.apply_chat_template(dialogs)`
    - `batch = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")`

- first, `apply_chat_template` make dialogs like this.
```
<|start_header_id|>user<|end_header_id|> What is shown in the image?
<|start_header_id|>assistant<|end_header_id|> This is a picture of a cat.
```

- and, `processor` will make image + text input for Llama 3.2 Vision Instruction
```python
batch = {
    "input_ids": tensor([[128006, 9125, 2178, ..., 0, 0, 0]]),          # text token
    "attention_mask": tensor([[1, 1, 1, ..., 0, 0, 0]]),                # mask for casual language model
    "pixel_values": tensor([[[[...]]]])                                 # image tensor
    'aspect_ratio_ids': tensor([[1]], device='cuda:0'),                 # image aspect ratio
    'aspect_ratio_mask': tensor([[[1, 0, 0, 0]]], device='cuda:0'),     # image aspect ratio massk
    'cross_attention_mask': tensor(...),                                # for image and language cross-attention
}
```

### Inference

```python
image = sample['images'][0].convert("RGB")  # if select first image.
dialog = sample['texts']

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"This is an image extracted using the Short-Time Fourier Transform (STFT) with librosa. The parameters used to extract this image are as follows: {dialog["feature_parameter"]}.\n {dialog["user_prompt"]}:"}
    ]}
]

inputs = processor(
    image,                      # no axis stft features
    input_text,                 # librosa feature parameters + Explanation about spectrogram
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
processor.decode(output[0])   # Explanation about spectrogram
```