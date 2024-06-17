from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json

from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

from tqdm import tqdm
import torch

from src.preprocessing import SemEval, GoEmotions
from src.utils import get_hf_token, write_jsonl
from src import model as M


@dataclass
class ScriptConfig:
    result_path: str = field(metadata={"help":"."})
    definitions_path: Optional[str] = field(default="data/prompts/raw_explanations/base_defs.json", metadata={"help":"."})
    prompt_path: Optional[str] = field(default="data/prompts/raw_explanations/base_prompt.txt", metadata={"help":"."})
    n_words: Optional[int] = field(default=20)
    # Generation

parser = HfArgumentParser([M.PretrainedLLMConfig, ScriptConfig])
llm_config, script_config = parser.parse_args_into_dataclasses()

# Load tokenizer and model

torch_dtype = torch.bfloat16
device_map = {"": 0}


tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name, padding_side="left", token=get_hf_token())
tokenizer.use_default_system_prompt = False
tokenizer.pad_token_id = tokenizer.eos_token_id


quantization_config = None
if llm_config.load_in_4bit or llm_config.load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=llm_config.load_in_8bit, 
        load_in_4bit=llm_config.load_in_4bit
    )
    

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_config.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    token=get_hf_token()
)

# Load correct data
_, examples = SemEval.load_data(return_sets="splits", urls=False, lower=False)
examples = examples.to_dict(orient='records')

definitions = json.loads(Path(script_config.definitions_path).read_text())
prompt = Path(script_config.prompt_path).read_text()

# Setup generation

generation_config = GenerationConfig(
  max_new_tokens=512,
  # max_length=4096,
  # do_sample=False,
  do_sample=True,
  temperature=.6,
  top_p=.9,
  eos_token_id=tokenizer.eos_token_id,
  pad_token_id=tokenizer.pad_token_id,
)

results = []

with torch.no_grad():
    for example in tqdm(examples):
        for i, definition in enumerate(definitions):
            
            example["n_words"] = script_config.n_words
            example["definition_id"] = i
            example["definition"] = definition

            turns = [{'role':'user', 'content':prompt.format(**example)}]
            input_ids = tokenizer.apply_chat_template(turns, return_tensors="pt").to(llm_model.device)

            outputs = llm_model.generate(
                input_ids,
                generation_config
            )
            
            example['question'] = tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip()
            example['answer'] = tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()

            results.append(dict(example))
            write_jsonl(script_config.result_path, results)

#  python -m src.raw_explanations --result_path="results/raw_explanations/llama3-8b_base_n20.jsonl" --model_name="meta-llama/Meta-Llama-3-8B-Instruct"