from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer
)

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import PeftModel
from datasets import Dataset

from torch import nn
from torch.utils.data import DataLoader
import torch
from src.model import MultiHeadCLM

from src.preprocessing import load_semeval_taskb, make_loader, preprocess_examples
from src.utils import get_hf_token, write_jsonl

import os

@dataclass
class ScriptArguments:
    
    # model
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})
    max_len: Optional[int] = field(default=105, metadata={"help":"drop example that have more token than max_len after tokenization"})
    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    
    # lora args
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapter"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapter"})
    peft_lora_bias: Optional[str] = field(default="none", metadata={"help":"use or not bias for the LoRA adapter"})
    peft_lora_dropout: Optional[float] = field(default=0.1, metadata={"help":"use or not dropout for the LoRA adapter"})

    # training args
    output_dir: Optional[str] = field(default="results/llama7b_chat_mh_hf", metadata={"help": "the output directory"})
    do_eval: Optional[bool] = field(default=True, metadata="whether to run evaluation on the validation set or not.")
    evaluation_strategy: Optional[str] = field(default="epoch", metadata="The evaluation strategy to adopt during training.")
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    learning_rate: Optional[float] = field(default=1.5e-4, metadata={"help": "the learning rate"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=10, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "Number of updates steps before checkpoint saves"})
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "Limits total number of checkpoints."})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata="whether or not to load the best model found during training at the end of training.")
    save_strategy: Optional[str] = field(default="epoch", metadata="The checkpoint save strategy to adopt during training.")
    
    # early stopping args
    early_stopping_patience: Optional[int] = field(default=5, metadata="stop training when the specified metric worsens for early_stopping_patience evaluation calls")
    early_stopping_threshold: Optional[float] = field(default=0.0, metadata="how much the specified metric must improve to satisfy early stopping conditions.")


script_args = ScriptArguments()

quantization_config = BitsAndBytesConfig(
    load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
)
torch_dtype = torch.bfloat16
device_map = {"": 0}

tokenizer = AutoTokenizer.from_pretrained(script_args.clm_model_name, padding_side="left", token=get_hf_token())
tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
tokenizer.use_default_system_prompt = False


clm_model = AutoModelForCausalLM.from_pretrained(
    script_args.clm_model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    token=get_hf_token()
)


clm_model.config.pad_token_id = tokenizer.pad_token_id
clm_model.resize_token_embeddings(len(tokenizer))


_, test = load_semeval_taskb(return_sets='splits', urls=False, lower=False)
test = preprocess_examples(tokenizer, test, script_args.max_len)
test_set = Dataset.from_list(test).map(lambda x: tokenizer(x['text']))


loader = make_loader(test_set, tokenizer, 1, extra_columns=True, shuffle=False)

    
model = MultiHeadCLM(clm_model)
model = PeftModel.from_pretrained(model, script_args.output_dir)
model = model.to(torch_dtype)
model.eval()

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
              'scores': scores.cpu().tolist(),
              'pred': scores.argmax(dim=1).cpu().item()
           }
        )

write_jsonl(script_args.output_dir + "/predictions.jsonl", results)
