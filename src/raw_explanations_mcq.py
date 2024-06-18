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
from src import model as M


@dataclass
class ScriptConfig:
    result_path: str = field(metadata={"help":"."})
    examples_path: Optional[str] = field(default="results/raw_explanations/llama3-8b_base_n20.jsonl", metadata={"help":"."})
    prompt_path: Optional[str] = field(default="data/prompts/raw_explanations/base_mcq_prompt.txt", metadata={"help":"."})
    # Generation

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
examples = defaultdict(list)
for example in read_jsonl(script_config.examples_path):
  examples[example['example_id']].append(example)


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

result_path = Path(script_config.result_path)
results = read_jsonl(result_path) if result_path.is_file() else []

statement_keys = "abcd"
answer_pattern = re.compile(r"\[([^\]]+)\]")


with torch.no_grad():
    for example in tqdm(examples.values()):

        if len(list(filter(lambda x: x['example_id'] == example[0]['example_id'], results))) == 1:
            print("skipped", example[0]['example_id'])
            continue

        statements = []
        for statement in example:
            s = answer_pattern.findall(statement["answer"])
            if len(s) == 0:
                s.append('')
            statements.append((statement["definition_id"], s[0]))

        new_example = {
            'example_id': example[0]['example_id'],
            'label_id': example[0]['label_id'],
            'text': example[0]['text'],
            'n_words': example[0]['n_words'],
        }

        for i in range(len(statements)):
            definition_ids, statement_texts = zip(*[(j, s) for j, s in statements[i:] + statements[:i]])

            new_example["definition_ids"] = definition_ids
            new_example["statements"] = '\n'.join([f"    {k}) {v}" for k, v in zip(statement_keys, statement_texts)])

            turns = [{'role':'user', 'content':prompt.format(**new_example)}]
            input_ids = tokenizer.apply_chat_template(turns, return_tensors="pt").to(llm_model.device)

            outputs = llm_model.generate(
                input_ids,
                generation_config
            )

            new_example['question'] = tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip()
            new_example['answer'] = tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()

            results.append(dict(new_example))
            write_jsonl(script_config.result_path, results)

#  python -m src.raw_explanations_mcq --result_path="results/raw_explanations/llama3-8b_base_n20_mcq.jsonl" --model_name="meta-llama/Meta-Llama-3-8B-Instruct"