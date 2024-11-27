For stable releases, use pip install unsloth. We recommend pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" for most installations though.

Conda Installation
⚠️Only use Conda if you have it. If not, use Pip. Select either pytorch-cuda=11.8,12.1 for CUDA 11.8 or CUDA 12.1. We support python=3.10,3.11,3.12.

```
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```



### Pip Installation

⚠️Do **NOT** use this if you have Conda. Pip is a bit more complex since there are dependency issues. The pip command is different for torch 2.2,2.3,2.4,2.5 and CUDA versions.

For other torch versions, we support torch211, torch212, torch220, torch230, torch240 and for CUDA versions, we support cu118 and cu121 and cu124. For Ampere devices (A100, H100, RTX3090) and above, use cu118-ampere or cu121-ampere or cu124-ampere.

For example, if you have torch 2.4 and CUDA 12.1, use:

```
pip install --upgrade pip
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

Another example, if you have torch 2.5 and CUDA 12.4, use:


```
pip install --upgrade pip
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```