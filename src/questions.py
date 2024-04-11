from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import random
import json


from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch

from src.utils import get_hf_token, read_jsonl, write_jsonl
from src.preprocessing import load_semeval_taskb
from src.model import load_mh

torch_dtype = torch.bfloat16
device_map = {"": 0}

@dataclass
class ScriptArguments:

    # model
    results_path: str = field(metadata={"help": "the path to save results"})
    clm_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})

    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})



parser = HfArgumentParser([ScriptArguments])
script_args = parser.parse_args_into_dataclasses()[0]


quantization_config = BitsAndBytesConfig(
   load_in_8bit=script_args.load_in_8bit, 
   load_in_4bit=script_args.load_in_4bit
)

# tokenizer, model = load_mh(
#     mh_model_name=script_args.mh_model_name,
#     clm_model_name=script_args.clm_model_name,
#     quantization_config=quantization_config,
#     torch_dtype=torch_dtype,
#     device_map=device_map,
#     hf_token=get_hf_token()
# )

tokenizer = AutoTokenizer.from_pretrained(script_args.clm_model_name, padding_side="left", token=get_hf_token())
tokenizer.use_default_system_prompt = False
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    script_args.clm_model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    token=get_hf_token()
)


def sample_texts(examples, n=5):
  return [example['text'] for example in random.sample(examples, n)]

def make_examples(examples, label_ids=[1,2], n=5):
  outputs = {}
  for i, label_id in enumerate(label_ids):
    outputs[f'set_{i}'] = sample_texts(list(filter(lambda x: x['label_id'] == label_id, examples)), n)
  return json.dumps(outputs, indent=2)


train, test = load_semeval_taskb(return_sets='splits', urls=False, lower=False)
train, test = train.to_dict(orient='records'), test.to_dict(orient='records')

questions_prompt = Path('data/prompts/questions/base.txt').read_text(encoding='utf-8')
consistency_prompt = Path('data/prompts/questions/consistency.txt').read_text(encoding='utf-8')

def generate_questions(model, prompt, examples, generation_config, label_ids=[1,2], n=5):

    random.shuffle(label_ids)

    set_0 = sample_texts(list(filter(lambda x: x['label_id'] == label_ids[0], examples)), n)
    set_1 = sample_texts(list(filter(lambda x: x['label_id'] == label_ids[1], examples)), n)

    results = {
        label_ids[0]: set_0,
        label_ids[1]: set_1,
        'true_set':label_ids[0],
        'false_set':label_ids[1],
    }

    model.eval()
    with torch.no_grad():

        sample = json.dumps({'set_0':set_0, 'set_1':set_1})

        turns = [{'role':'user', 'content':prompt.format(examples=sample).strip()}]
        input_ids = tokenizer.apply_chat_template(turns, return_tensors="pt").to(model.device)

        outputs = model.generate(
            input_ids,
            attention_mask=torch.full(input_ids.shape, 1),
            generation_config=generation_config
        )

        results['questions_prompt'] = tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip()
        results['questions_answer'] = tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()

    return results

import re
json_pattern = re.compile(r"\`\`\`json(.+)\`\`\`", re.S)

def parse_answer(answer):
    return json.loads(json_pattern.search(answer).group(1))

def consistency_check(model, prompt, questions, generation_config, label_ids=[1,2]):

    results = []

    try:
        qlist = parse_answer(questions['questions_answer'])
    except Exception as e:
        print(e)
        return questions

    examples = questions[label_ids[0]] + questions[label_ids[1]]

    logits_ids = torch.tensor([5574,  8824,  2089,  1565, 17131, 25717, 20652,  4541,  7700, 15676, 3009,  5852]).to(model.device)
    act_func = torch.nn.Softmax(dim=-1)

    model.eval()
    with torch.no_grad():
        for q in qlist:
            for example in examples:
                turns = [{'role':'user', 'content':prompt.format(text=example, question=q).strip()}]
                input_ids = tokenizer.apply_chat_template(turns, return_tensors="pt").to(model.device)

                outputs = model.forward(
                    input_ids,
                    attention_mask=torch.full(input_ids.shape, 1)
                )

                scores = act_func(outputs['logits'][..., -1, logits_ids])

                outputs = model.generate(
                    input_ids,
                    attention_mask=torch.full(input_ids.shape, 1),
                    generation_config=generation_config
                )

                results.append({
                    **questions,
                    'question': q,
                    'example': example,
                    'scores': scores.tolist(),
                    'pred': scores.argmax().item(),
                    'consistency_prompt': tokenizer.decode(outputs[0, :input_ids.shape[1]]).strip(),
                    'consistency_answer': tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()
                })
    return results

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6, # lower
    top_p=0.9, # higher
    top_k=50, # lower
    repetition_penalty=1.2, # lower
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

results = []

for i in tqdm(range(100)):

    outputs = generate_questions(
        model=model, 
        prompt=questions_prompt, 
        examples=train, 
        generation_config=generation_config
    )

    results.extend(consistency_check(
        model=model,
        prompt=consistency_prompt,
        questions=outputs,
        generation_config=generation_config
    ))

    write_jsonl(script_args.results_path, results)