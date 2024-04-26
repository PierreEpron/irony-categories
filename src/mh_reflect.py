from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import json

from transformers import (
    HfArgumentParser,
    GenerationConfig,
    BitsAndBytesConfig,
)

from datasets import Dataset
import torch

from src.preprocessing import load_semeval_taskb, make_loader, preprocess_examples, format_labeled_turns
from src.utils import get_hf_token, read_jsonl, write_jsonl
from src.model import load_mh


torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:

    # model
    mh_model_name: str = field(metadata={"help": "the directory used to load mh_model"})
    clm_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})
    max_len: Optional[int] = field(default=105, metadata={"help":"drop example that have more token than max_len after tokenization"})

    # prompt
    expl_prompt_path: Optional[str] = field(default="data/prompts/reflects/expl.json", metadata={"help":"the path used to load explanation prompts."})
    refl_prompt_path: Optional[str] = field(default="data/prompts/reflects/refl.json", metadata={"help":"the path used to load reflection prompts."})

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
model.eval()


_, test = load_semeval_taskb(return_sets='splits', urls=False, lower=False).to_dict(orient='records')
expl_prompts = json.loads(Path(script_args.expl_prompt_path).read_text())
refl_prompts = json.loads(Path(script_args.refl_prompt_path).read_text())


generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)


results_path = Path(script_args.mh_model_name + "/" + "reflects.jsonl")
results = read_jsonl(results_path) if results_path.is_file() else []

act_func = torch.nn.Softmax(dim=1)

def format_turns(turns, example):
  return [{'role':turn['role'], 'content':turn['content'].format(**example)} for turn in turns]

with torch.no_grad():
  for example in tqdm(test):

    if len(list(filter(lambda x: x['example_id'] == example['example_id'], results))) != 0:
        continue

    example = {
      'example_id': example['example_id'],
      'label_id': example['label_id'],
      'text': example['text']
    }

    input_ids = tokenizer.apply_chat_template(
       [{'role':'user', 'content':example['text']}], 
       return_tensors='pt'
    ).to(model.device)

    attention_mask = torch.LongTensor([[1] * input_ids.shape[0]]).to(model.device)

    label_id = torch.LongTensor([[example['label_id']]]).to(model.device)

    outputs = model(
       input_ids=input_ids, 
       attention_mask=attention_mask,
       label_id=label_id
    )

    scores = act_func(outputs.logits)
    pred = scores.argmax(dim=1).cpu().item()

    example.update({
      'scores': scores.cpu().tolist(),
      'pred': pred
    })

    for k, turns in expl_prompts:

        input_ids = tokenizer.apply_chat_template(
           format_turns(turns, example), 
           return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(input_ids, generation_config)

    results.append(example)

    write_jsonl(results_path, results)