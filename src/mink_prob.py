from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModel
)
import torch

# from src.preprocessing import load_data
from src.preprocessing import load_semeval_taskb
from src.utils import get_hf_token, write_jsonl
from src import model as M

@dataclass
class ScriptConfig:
    result_path: Optional[str] = field(metadata={"help":"."})
    k: Optional[int] = field(default=20, metadata={"help":"."})


parser = HfArgumentParser([M.PretrainedLLMConfig, ScriptConfig])
llm_config, script_config = parser.parse_args_into_dataclasses()

# Loading model could be shared with LLMClassifier.load_llm

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
    

llm_model = AutoModel.from_pretrained(
    llm_config.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    token=get_hf_token()
)


data = load_semeval_taskb(
    return_sets="full",
    hashtag_labels=False,
    hashtag_nots=False,
    users=False,
    urls=False,
    spaces=False,
    lower=False
).to_dict(orient='records')


def mink_prob(scores, k=20):

    # Compute length of Min-K set.
    ratio = k / 100 # TODO: if k is intance of float it's already a ratio.
    k_length = int(len(scores)*ratio)

    # Recover the Min-K set by taking the lowest scores.
    topk_prob = gold_scores.sort()[0][:k_length]

    # Compute and return the average.
    return -topk_prob.mean().item()

with torch.no_grad():
    for example in tqdm(data):
        input_ids = tokenizer.encode(example['text'], return_tensors='pt').to(llm_model.device)
        outputs = llm_model(input_ids, return_dict=True)

        logits = outputs['logits']

        scores = torch.nn.functional.log_softmax(logits, dim=-1)
        gold_scores = scores[0, torch.arange(input_ids.shape[-1]-1), input_ids[0][1:]]

        example['mink_prob'] = mink_prob(gold_scores, script_config.k)

write_jsonl(script_config.result_path, data)

