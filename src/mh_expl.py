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
from src.utils import get_hf_token, write_jsonl
import warnings

torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:

    # model
    mh_model_name: str = field(metadata={"help": "the directory used to load mh_model"})
    clm_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})
    max_len: Optional[int] = field(default=105, metadata={"help":"drop example that have more token than max_len after tokenization"})


    # prompt
    prompt_path: Optional[str] = field(default="data/prompts", metadata={"help":"the path used to load prompt(s). If path is a json file, load prompt from it. If path is a dir, try to execute each json file inside as a prompt"})
    expl_from_gold: Optional[bool] = field(default=True, metadata={"help":"If true, will apply all the prompt taking into account the gold label."})
    expl_from_pred: Optional[bool] = field(default=True, metadata={"help":"If true, will apply all the prompt taking into account the pred label."})


    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})


parser = HfArgumentParser([ScriptArguments])
script_args = parser.parse_args_into_dataclasses()[0]

if not (script_args.expl_from_gold or script_args.expl_from_pred):
    warnings.warn(f"One of expl_from_gold ({script_args.expl_from_gold}) or expl_from_pred ({script_args.expl_from_pred}) should be set to True. Otherwise no explaination prompt is used.")


parser = HfArgumentParser([ScriptArguments])
script_args = parser.parse_args_into_dataclasses()[0]


if not (script_args.expl_from_gold or script_args.expl_from_pred):
    warnings.warn(f"One of expl_from_gold ({script_args.expl_from_gold}) or expl_from_pred ({script_args.expl_from_pred}) should be set to True. Otherwise no explaination prompt is used.")


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


prompt_path = Path(script_args.prompt_path)
if prompt_path.is_dir():
    prompts = {path.stem:json.loads(path.read_text()) for path in prompt_path.glob('*.json')}
else:
    prompts = {prompt_path.stem:json.loads(prompt_path.read_text())}


generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    top_k=60,
    repetition_penalty=1.2
)


results = []
act_func = torch.nn.Softmax(dim=1)

with torch.no_grad():
  for batch in loader:

    gold = batch['label_id'][0].cpu().item()
    text = batch['text'][0]

    example = {
      'example_id': batch['example_id'][0],
      'label_id': gold,
      'text': text
    }

    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], label_id=batch['label_id'])
    scores = act_func(outputs.logits)
    pred = scores.argmax(dim=1).cpu().item()

    example.update({
      'scores': scores.cpu().tolist(),
      'pred': pred,
      'gold_expls': {},
      'pred_expls': {}
    })

    text = text.replace('<s> [INST]', '').replace('[/INST]', '').strip()

    for k, prompt in prompts.items():
        input_ids = format_labeled_turns(tokenizer, gold, prompt, {'text':text})
        outputs = model.generate(input_ids, generation_config)
        example['gold_expls'][k] = tokenizer.decode(outputs[0])

    for k, prompt in prompts.items():
        input_ids = format_labeled_turns(tokenizer, pred, prompt, {'text':text})
        outputs = model.generate(input_ids, generation_config)
        example['pred_expls'][k] = tokenizer.decode(outputs[0])

    results.append(example)

    write_jsonl("explanations.jsonl", results)