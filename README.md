# Large Anomaly Detection Language Model (LADLM)

##### checkout: https://github.com/UTurtle/anomaly

### DCASE 2024 Task 2: First-Shot Unsupervised Anomalous Sound Detection

How to easy EDA (Exploratory Data Analysis)? -> using Multi-modal LLM


---

### How to setting

1. make new python enviroment(or conda) `conda create -n ladlm python=3.11 -y`
2. set your new enviroment `conda activate ladlm`
3. `conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
4. `pip install transformers datasets`

---





#### TODO

- [ ] git init . and add . and commit -m "first commit" and git push origin main

- just using unsloth (for easy training)
    - try notebook `Llama_3_2_Vision_Finetuning_Unsloth_Radiography`.


- __Testing Llama Vison Model__ (current works)
    - Llama 3.2 Vision Model Install
        - other model in [Llama huggingface](https://huggingface.co/meta-llama)
        - we select pretrained model: [Llama 3.2 Vision Instruction 11B huggingface](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
    - Llama 3.2 Vision Model Fine-tuning (PEFT) in notebook
        - dataset: [OCRVQA-dataset](https://ocr-vqa.github.io/)
        - task : ocr-image
        - we using peft: 16vram is not enough for vision llm.
          - QLoRA is good solution for fine-tuning for 16g vram
        - [x] using unsloth
        - [x]  using llama_recipe


  - __dataset preprocessing for Llama 3.2 11B Vision Model PEFT__
    - [x] learn Llama 3.2 11B Vision Model input format.
    - [x] Make big feature json [anomaly feature json](https://github.com/UTurtle/anomaly/blob/main/eda/
    - [x] learn ChartQA vison input format for best training enviroment!
        - [ChartQA](https://arxiv.org/abs/2203.10244)
            - image input format: (`224x224` ~ `512x512` but it's good?)
    - [x] discussion input dataset.
    - [x] Make big feature json [anomaly feature json](https://github.com/UTurtle/anomaly/blob/main/eda/extract_feature_code/audio_features.json)
    - [x] make script `ladlm_dataset.py` extracting from json. referencing `ocrvqa_dataset.py`.


- __Try Llama 3.2 11B Vision Model PEFT using dataset ladlm__
    - [x] re-script llama_recpe peft notebook for ladlm_dataset
    - [x] evaluation first peft
      - [ ] human evaluation
      - [x] gpt4o mini evaluation


 - __Augmentation spectrogram:eda-explain-text pair data automation__
   - [ ] scripting spectrogram_stft_maker
     - [x] Add horizon and vertical sprite pattern maker
     - [x] Add noise whole audio (can select)
       - [x] gaussian noise 
       - [x] linear noise
       - [x] others..
     - [x] Add shape noise (can select)
     - [x] Add eda-explain-text template
     - [ ] Automation spectrogram generator 
       - [x] Make automation data pipeline sample
       - [x] Refactor ipynb to py
       - [ ] to create more Generalization data (Lack of detail)
       - [ ] make for level system for Scheduling Data (just show complex level!)
       - [ ] condense json for secure resources
     - [ ] Make Dataset 
     - [ ] PEFT


- __PEFT Large Anomaly Detection Language Model__ 
    - [ ] build baseline model

 
---

#### directory structure

```
ladlm
├── README.md
├── checkpoints
│   └── baseline
├── datasets
│   └── preprocessed
├── notebook (just try vision llm)
│   └── llama_recipe
│   ├── unsloth
├── model explain.md
├── peft
│   └── baseline.py
├── preprocessing
│   └── baseline.py
├── run.py
└── test
    └── baseline.py
```

##### Reference

- [dcase anomaly detection](https://github.com/UTurtle/anomaly)
- [llama-recipe](https://github.com/meta-llama/llama-recipes/tree/main)
- [llama3.2-vision-finetuning](https://github.com/2U1/Llama3.2-Vision-Finetune/tree/master)