from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)

from peft import PeftModel
from datasets import Dataset

import torch
from src.model import MultiHeadCLM, load_mh

from src.preprocessing import load_semeval_taskb, make_loader, preprocess_examples, format_labeled_turns
from src.utils import get_hf_token, read_jsonl, write_jsonl
import warnings

torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:

    # model
    mh_model_name: str = field(metadata={"help": "the directory used to load mh_model"})
    clm_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})
    
    # prompt
    prompt_path: Optional[str] = field(default="data/prompts", metadata={"help":"the path used to load prompt(s). If path is a json file, load prompt from it. If path is a dir, try to execute each json file inside as a prompt"})

    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})

    # generation
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "max new token for generation config"})
    

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
model.eval()


_, examples = load_semeval_taskb(return_sets='splits', urls=False, lower=False)
examples = examples.to_dict(orient='records')


def format_turns(prompt, example):
  return [{'role':'user', 'content':prompt.format(**example).strip()}]

prompt = Path(script_args.prompt_path).read_text(encoding="utf-8")

generation_config = GenerationConfig(
    max_new_tokens=script_args.max_new_tokens,
    do_sample=False
)


results_path = Path(script_args.mh_model_name + "/" + f"{Path(script_args.prompt_path).stem}.jsonl")
results = read_jsonl(results_path) if results_path.is_file() else []

act_func = torch.nn.Softmax(dim=1)

with torch.no_grad():
  for example in examples:

    if len(list(filter(lambda x: x['example_id'] == example['example_id'], results))) != 0:
        continue

    input_ids = tokenizer.apply_chat_template(format_turns(prompt, example), return_tensors="pt").to(model.clm_model.device)

    outputs = model.generate(
        input_ids,
        generation_config
    )

    example['prompt'] = tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip()
    example['answer'] = tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()

    results.append(example)

    write_jsonl(results_path, results)