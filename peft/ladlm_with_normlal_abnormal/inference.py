import matplotlib.pyplot as plt
import torch


def inference_vllm(model, processor, sample, max_token=512):
    # Extract and convert images
    linear_spectrogram_with_axes_image = sample['linear_spectrogram_with_axes']['image'].convert("RGB") # we can use this?
    linear_spectrogram_no_axes_image = sample['linear_spectrogram_no_axes']['image'].convert("RGB")
    
    machine_type = sample['machineType']
    librosa_parameters = sample['linear_spectrogram_no_axes']['librosa_parameters']
    messages = [
        [
            {
                "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": f"This is an image extracted from machine({machine_type}) noise using the Short-Time Fourier Transform (STFT) with librosa."
                                 f"The parameters used to extract this image are as follows: {librosa_parameters}. "
                                 f"\nExplanation about spectrogram and Detection anomaly:\n"}
                ]
            },
        ]
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        linear_spectrogram_no_axes_image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_token)
    generated_text = processor.decode(output[0])
    plt.imshow(linear_spectrogram_with_axes_image)
    plt.axis('off')
    plt.show()
    return generated_text
