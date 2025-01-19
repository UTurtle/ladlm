import matplotlib.pyplot as plt
import torch


def inference_vllm(model, processor, sample, max_token=512):
    
    print(sample['spectrogram_with_axes'].convert("RGB"))
    spectrogram_with_axes = sample['spectrogram_with_axes'].convert("RGB")
    spectrogram_no_axes = sample['spectrogram_no_axes'].convert("RGB")
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "This is an image extracted using the Short-Time Fourier Transform (STFT) with librosa.\nspectrogram json:\n"}
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
    generated_text = processor.decode(output[0])
    plt.imshow(spectrogram_with_axes)
    plt.axis('off')
    plt.show()
    return generated_text
