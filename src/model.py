from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights
import torch

from src.utils import get_hf_token

def load_clm(model_name, empty_weight=False):
    
    if empty_weight:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=get_hf_token()
        )

    else:
        torch_dtype = torch.bfloat16
        device_map = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=get_hf_token()
        )

    return model