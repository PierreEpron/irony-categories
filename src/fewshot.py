from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import json

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    GenerationConfig,
    BitsAndBytesConfig,
)
import torch

from src.preprocessing import load_semeval_taskb
from src.utils import get_hf_token, write_jsonl

##### Script args #####

torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:
  # prompt
  prompt_path: str = field(metadata={"help":"the path used to load prompt. Prompt should be in turn format (json). All turn content will be formated using example"})
  output_path: str = field(metadata={"help":"the path used to save outputs."})

  # labels 
  target_labels: Optional[List[str]] = field(default=[0,1,2,3], metadata={"help":"the examples used will first filter using target_labels."})
  logits_labels: Optional[List[str]] = field(default='yes no Yes No', metadata={"help":"a string that will be encoded by the tokenizer. Token IDs will be used to pre-select logits to score."})

  # model
  clm_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})

  # generation args  

  max_new_tokens: Optional[int] = field(default=16, metadata={"help": "maximum of new tokens to generate"}),
  do_sample: Optional[int] = field(default=False, metadata={"help": "use or not a sampling decoder"}),

  # b&b args
  load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
  load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})

parser = HfArgumentParser([ScriptArguments])
script_args = parser.parse_args_into_dataclasses()[0]

##### Tokenizer & Model #####

tokenizer = AutoTokenizer.from_pretrained(script_args.clm_model_name, padding_side="left", token=get_hf_token())
tokenizer.use_default_system_prompt = False

quantization_config = BitsAndBytesConfig(
  load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
)

clm_model = AutoModelForCausalLM.from_pretrained(
  script_args.clm_model_name,
  quantization_config=quantization_config,
  device_map=device_map,
  torch_dtype=torch_dtype,
  token=get_hf_token()
)

tokenizer.pad_token_id = tokenizer.eos_token_id
clm_model.config.pad_token_id = tokenizer.pad_token_id

##### Examples #####

def format_turns(turns, example):
  return [{'role':turns['role'], 'content':turns['content'].format(**example)}]

examples = load_semeval_taskb(return_sets='full', urls=False, lower=False)
examples = examples.to_dict(orient='records')

label_ids = tokenizer.encode(
   script_args.logits_labels, 
   add_special_tokens=False, 
   return_tensors='pt'
)
label_ids = label_ids[0].to(clm_model.device)

generation_config = GenerationConfig(
    max_new_tokens=script_args.max_new_tokens,
    do_sample=script_args.do_sample,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

##### Turns #####

turns = json.loads(Path(script_args.prompt_path).read_text())

##### Loop #####

results = []
sm = torch.nn.Softmax(dim=-1)

torch.cuda.empty_cache()
clm_model.eval()

with torch.no_grad():
  for example in tqdm(examples):
      
      if example['label_id'] not in script_args.target_labels:
        continue

      ##### Encode examples #####

      input_ids = torch.tensor([format_turns(turns, example)]).to(clm_model.device)

      ##### Generate answer (opened anwser) #####

      outputs = clm_model.generate(
        input_ids,
        generation_config
      )
      example['answer'] = tokenizer.decode(outputs[0, input_ids.shape[1]:])

      ##### Argmax answer (closed anwser) #####

      scores = sm(clm_model.forward(input_ids)['logits'][0, -1, label_ids])
      example['scores'] = sm(scores).tolist()
      example['pred'] = sm(scores).argmax().item()

      results.append(example)

      write_jsonl(script_args.output_path, results)