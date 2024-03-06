# -*- coding: utf-8 -*-

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
from peft import LoraConfig, get_peft_model
from datasets import Dataset

from torch import nn
import torch

from src.model import load_mh
from src.preprocessing import load_semeval_taskb, make_loader, preprocess_examples, collate
from src.utils import get_hf_token, write_jsonl


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


torch_dtype=torch.bfloat16 # TODO: Find a way to put it in args
device_map={"":0} # TODO: Find a way to put it in args


# TODO: Save args
script_args = ScriptArguments()


##### Load data & compute class weight #####


train, test = load_semeval_taskb(return_sets='splits', urls=False, lower=False)
label_weigths = torch.tensor((train.label_id.value_counts() / len(train)).values).to(dtype=torch_dtype)


##### Load/Create Model #####
    

quantization_config = BitsAndBytesConfig(
   load_in_8bit=script_args.load_in_8bit, 
   load_in_4bit=script_args.load_in_4bit
)

lora_config = LoraConfig(
    r=script_args.peft_lora_alpha,
    lora_alpha=script_args.peft_lora_r,
    target_modules=["q_proj", "v_proj"], # TODO: Find a way to put it in args
    lora_dropout=script_args.peft_lora_dropout,
    bias=script_args.peft_lora_bias,
    modules_to_save=["score"], # TODO: Find a way to put it in args
)

tokenizer, model = load_mh(
   clm_model_name=script_args.model_name,
   quantization_config=quantization_config,
   lora_config=lora_config,
   label_weigths=label_weigths,
   torch_dtype=torch_dtype,
   device_map=device_map,
   hf_token=get_hf_token()
)


##### Data Preps #####


# TODO: Find better way
train, test = preprocess_examples(tokenizer, train, script_args.max_len), preprocess_examples(tokenizer, script_args.max_len)

# TODO: Shuffle for multiple runs
train_set = Dataset.from_list(train[len(test):]).map(lambda x: tokenizer(x['text']))
val_set = Dataset.from_list(train[:len(test)]).map(lambda x: tokenizer(x['text']))
test_set = Dataset.from_list(test).map(lambda x: tokenizer(x['text']))

print(len(train_set), len(val_set), len(test_set))


##### Training #####


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    do_eval=script_args.do_eval,
    evaluation_strategy=script_args.evaluation_strategy,
    per_device_train_batch_size=script_args.batch_size,
    per_device_eval_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    load_best_model_at_end=script_args.load_best_model_at_end,
    save_strategy=script_args.save_strategy,
    remove_unused_columns=False,
    label_names=["label_id"],
)

early_stop = EarlyStoppingCallback(
    script_args.early_stopping_patience,
    script_args.early_stopping_threshold
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate(tokenizer),
    train_dataset=train_set,
    eval_dataset=val_set,
    callbacks=[early_stop]
)


trainer.train()
trainer.save_model(training_args.output_dir)
trainer.save_state()


##### Predictions #####


loader = make_loader(test, tokenizer, 1, extra_columns=True, shuffle=False)

model = trainer.model
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
