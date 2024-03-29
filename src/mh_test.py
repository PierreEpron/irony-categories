from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel
from datasets import Dataset

import torch
from src.model import MultiHeadCLM, load_mh

from src.preprocessing import load_semeval_taskb, make_loader, preprocess_examples
from src.utils import get_hf_token, write_jsonl


torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:

    # model
    mh_model_name: str = field(metadata={"help": "the directory used to load mh_model"})
    
    clm_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})
    max_len: Optional[int] = field(default=105, metadata={"help":"drop example that have more token than max_len after tokenization"})
    
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

_, test = load_semeval_taskb(return_sets='splits', urls=False, lower=False)
test = preprocess_examples(tokenizer, test, script_args.max_len)
test_set = Dataset.from_list(test).map(lambda x: tokenizer(x['text']))

loader = make_loader(test_set, tokenizer, 1, extra_columns=True, shuffle=False)

act_func = torch.nn.Softmax(dim=1)

results = []

with torch.no_grad():
    for batch in tqdm(loader):
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], label_id=batch['label_id'])
        scores = act_func(outputs.logits)
        results.append(
           {
              'example_id': batch['example_id'][0],
              'label_id': batch['label_id'][0].cpu().item(),
              'text': batch['text'][0],
              'scores': scores.cpu().tolist(),
              'pred': scores.argmax(dim=1).cpu().item()
           }
        )

write_jsonl(script_args.mh_model_name + "/predictions.jsonl", results)
