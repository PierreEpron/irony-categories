from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from pathlib import Path
import re

from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

from tqdm import tqdm
import torch

from src.utils import get_hf_token, read_jsonl, write_jsonl
from src.preprocessing import SemEval
from src import model as M


@dataclass
class ScriptConfig:
    result_path: str = field(metadata={"help":"."})
    prompt_path: Optional[str] = field(metadata={"help":"."})
    # Generation
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"}),
    do_sample: Optional[bool] = field(default=True, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"}),
    temperature: Optional[float] = field(default=0.6, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"}),
    top_p: Optional[float] = field(default=0.9, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"}),

parser = HfArgumentParser([M.PretrainedLLMConfig, ScriptConfig])
llm_config, script_config = parser.parse_args_into_dataclasses()


# Load tokenizer and model


torch_dtype = torch.bfloat16
device_map = "auto"

tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name, padding_side="left", token=get_hf_token())
tokenizer.use_default_system_prompt = False
tokenizer.eos_token_id = tokenizer.encode("<|eot_id|>")[-1]
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

prompt = Path(script_config.prompt_path).read_text()


# Setup generation


generation_config = GenerationConfig(
  max_new_tokens=script_config.max_new_tokens,
  do_sample=script_config.do_sample,
  temperature=script_config.temperature,
  top_p=script_config.top_p,
  eos_token_id=tokenizer.eos_token_id,
  pad_token_id=tokenizer.pad_token_id,
)

result_path = Path(script_config.result_path)
results = read_jsonl(result_path) if result_path.is_file() else []

with torch.no_grad():
    for example in examples:

        if len(list(filter(lambda x: x['example_id'] == example['example_id'], results))) >= 1:
            print("skipped", example['example_id'])
            continue

        turns = [{'role':'user', 'content':prompt.format(**example)}]
        input_ids = tokenizer.apply_chat_template(turns, return_tensors="pt").to(llm_model.device)

        outputs = llm_model.generate(
            input_ids,
            generation_config
        )

        example['question'] = tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip()
        example['answer'] = tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()

        results.append(example)
        write_jsonl(script_config.result_path, results)

#  python -m src.generate --result_path="results/llama3-8b_json-free.jsonl" --prompt_path="data/prompts/json/free.txt" --model_name="meta-llama/Meta-Llama-3-8B-Instruct" --max_new_tokens=1024