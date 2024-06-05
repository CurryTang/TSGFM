#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from graphllm.language_model.llaga_mistral import LlagaMistralForCausalLM
from graphllm.constants import DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from huggingface_hub import hf_hub_download




def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", cache_dir="../../checkpoint"):
    kwargs = {"device_map": device_map}


    kwargs['torch_dtype'] = torch.float16

    if 'llaga' in model_name.lower():
        if model_base is not None:
            # this may be mm projector only
            print('Loading LLaGA from base model...')
            if 'mistral' in model_base:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlagaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, cache_dir=cache_dir,
                                                              **kwargs)
            else:
                raise NotImplementedError("Only Mistral is supported for now")
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                print("Load from local path")
            else:
                from huggingface_hub import hf_hub_download
                model_path_hf = hf_hub_download(repo_id=model_path,  filename='mm_projector.bin')
                mm_projector_weights = torch.load(model_path_hf, map_location='cpu')
                print("Load from huggingface")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlagaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            # from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, cache_dir=cache_dir, **kwargs)


    if 'llaga' in model_name.lower():
        mm_use_graph_start_end = getattr(model.config, "mm_use_graph_start_end", False)
        if mm_use_graph_start_end:
            tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len