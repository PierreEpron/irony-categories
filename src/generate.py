from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

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
from src.postprocessing import VALID_MAP
from src import preprocessing as P
from src import model as M

@dataclass
class ScriptConfig:
    result_path: str = field(metadata={"help":"."})
    sharded: Optional[bool] = field(default=False, metadata={"help":"."})
    max_try: Optional[int] = field(default=10, metadata={"help":"."})
    valid_answer: Optional[str] = field(default=None, metadata={"help":"[None, 'text', 'json']"})
    # Generation
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"})
    do_sample: Optional[bool] = field(default=True, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"})
    temperature: Optional[float] = field(default=0.6, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"})
    top_p: Optional[float] = field(default=0.9, metadata={"help": "see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig"})

parser = HfArgumentParser([M.PretrainedLLMConfig,  P.DataConfig, ScriptConfig])
llm_config, data_config, script_config = parser.parse_args_into_dataclasses()

# Load tokenizer and model

torch_dtype = torch.bfloat16
device_map = "auto" if script_config.sharded else {"":0}

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
llm_model.eval()


##### Load and preprocess data #####

if data_config.dataset not in P.MANAGER_CLASS_MAP:
    raise AttributeError(f"`data_config.dataset` should be equal to ['semeval', 'goemotions'] not to {data_config.dataset}")

data_manager = P.MANAGER_CLASS_MAP[data_config.dataset](tokenizer, data_config)
train_examples, val_examples, test_examples = data_manager.process_data()
examples = train_examples.to_list() + val_examples.to_list() + test_examples.to_list()

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
    for example in tqdm(examples):
        for i in range(script_config.max_try):

            if len(list(filter(lambda x: x['example_id'] == example['example_id'], results))) >= 1:
                print("skipped", example['example_id'])
                continue
            
            input_ids = torch.tensor([example['input_ids']]).to(device=llm_model.device, dtype=torch.long)

            outputs = llm_model.generate(
                input_ids,
                generation_config
            )

            example['question'] = tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip()
            example['answer'] = tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()

            if script_config.valid_answer and VALID_MAP[script_config.valid_answer](example['answer']):
                results.append(example)
                write_jsonl(script_config.result_path, results)
                break
            else:
                print(example['answer'])

print(len(results), "/", len(examples))

#  python -m src.generate --result_path="results/llama3-8b_json-free.jsonl" --prompt_path="data/prompts/json/free.txt" --model_name="meta-llama/Meta-Llama-3-8B-Instruct" --max_new_tokens=1024