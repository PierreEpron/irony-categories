from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    GenerationConfig,
    BitsAndBytesConfig,
)
import torch

from src.utils import get_hf_token
from src.model import load_mh


torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:

    # model
    mh_model_name: str = field(metadata={"help": "the directory used to load mh_model"})
    clm_model_name: str = field(metadata={"help": "the model name"})

    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})


parser = HfArgumentParser([ScriptArguments])
script_args = parser.parse_args_into_dataclasses()[0]

quantization_config = BitsAndBytesConfig(
    load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
)

tokenizer, model = load_mh(
    mh_model_name=script_args.mh_model_name,
    clm_model_name=script_args.clm_model_name,
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
    device_map=device_map,
    hf_token=get_hf_token()
)

print(model)
