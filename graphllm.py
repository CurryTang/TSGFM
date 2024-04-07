# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
from graphllm.train import _train

## three modes:
## 1. pre-train
## 2. eval pre-train
## 3. eval res





if __name__ == "__main__":
    _train()