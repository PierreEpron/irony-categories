from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from transformers import (
    HfArgumentParser,
    GenerationConfig,
    BitsAndBytesConfig,
)


import torch
from src.model import load_mh


from src.preprocessing import load_semeval_taskb, format_turns
from src.postprocessing import is_valid_enum

from src.utils import get_hf_token, read_jsonl, write_jsonl


torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:

    # model
    mh_model_name: str = field(metadata={"help": "the directory used to load mh_model"})
    clm_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})

    # prompt
    prompt_path: Optional[str] = field(default="data/prompts", metadata={"help":"the path used to load prompt(s). If path is a txt file, load prompt from it. If path is a dir, try to laod each txt file inside as a prompt"})
    max_try: Optional[int] = field(default=10, metadata={"help":"the number of maximum try to try to generate a correct answer"})

    # results
    results_path: Optional[str] = field(default="results/enum_expl.jsonl", metadata={"help":"the path used to store results"})
 
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


_, examples = load_semeval_taskb(return_sets='splits', urls=False, lower=False)
examples = examples.to_dict(orient='records')


prompt_path = Path(script_args.prompt_path)
if prompt_path.is_dir():
    prompts = {path.stem:path.read_text(encoding='utf-8') for path in prompt_path.glob('*.txt')}
else:
    prompts = {prompt_path.stem:prompt_path.read_text(encoding='utf-8')}


generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.3, # lower
    top_p=0.95, # higher
    top_k=25, # lower
    repetition_penalty=1.2 # lower
)

results_path = Path(script_args.results_path)
results = read_jsonl(results_path) if results_path.is_file() else []

model.eval()
with torch.no_grad():
  for example in tqdm(examples):
    for k_prompt, prompt in prompts.items():

        if len(list(filter(lambda x: x['example_id'] == example['example_id'] and x['prompt_key'] == example['prompt_key'], results))) != 0:
            continue

        anwser = ''
        n_try = 0

        while not is_valid_enum(anwser) and n_try < script_args.max_try: #TODO: This should be configurable. Here it's work only for enum.

            turns = format_turns(prompt, example)
            input_ids = tokenizer.apply_chat_template(turns, return_tensors="pt").to(model.device)
            outputs = model.generate(input_ids, generation_config)
            
            answer = tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip()
            n_try += 1

            
        results.append({
            **example,
            'prompt_key': k_prompt,
            'n_try':n_try,
            'prompt': tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip(),
            'answer': answer,
            'n_try':n_try,
        })

        write_jsonl(results_path, results)