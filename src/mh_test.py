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

from src.preprocessing import load_semeval_taskb
from src.utils import get_hf_token

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@dataclass
class ScriptArguments:
    # model
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf" , metadata={"help": "the model name"})
    max_len: Optional[int] = field(default=100, metadata={"help":"drop example that have more token than max_len after tokenization"})
    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    # lora args
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_bias: Optional[str] = field(default="none", metadata={"help":"..."}),
    peft_dropout: Optional[float] = field(default=0.1, metadata={"help":"..."}),

    # training args
    output_dir: Optional[str] = field(default="results/llama7b_chat_mh_cls", metadata={"help": "the output directory"})
    do_eval: Optional[bool] = field(default=True, metadata="whether to run evaluation on the validation set or not.")
    evaluation_strategy: Optional[str] = field(default="epoch", metadata="The evaluation strategy to adopt during training.")
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=10, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "Limits total number of checkpoints."})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata="whether or not to load the best model found during training at the end of training.")
    save_strategy: Optional[str] = field(default="epoch", metadata="The checkpoint save strategy to adopt during training.")
    # early stopping args
    early_stopping_patience: Optional[int] = field(default=5, metadata="stop training when the specified metric worsens for early_stopping_patience evaluation calls")
    early_stopping_threshold: Optional[float] = field(default=0.0, metadata="how much the specified metric must improve to satisfy early stopping conditions.")

script_args = ScriptArguments(batch_size=4, max_len=105)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
)
torch_dtype = torch.bfloat16
device_map = {"": 0}

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left", token=get_hf_token())
tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
tokenizer.use_default_system_prompt = False

clm_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    token=get_hf_token()
)

clm_model.config.pad_token_id = tokenizer.pad_token_id
clm_model.resize_token_embeddings(len(tokenizer))

class MultiHeadCLM(nn.Module):
    def __init__(self, clm_model, context_size=4096, num_labels=4, label_weigths=None) -> None:
        super().__init__()

        self.clm_model = clm_model

        self.context_size = context_size
        self.num_labels = num_labels
        self.label_weights=label_weigths

        self.score = nn.Linear(context_size, out_features=num_labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        label_id: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        # print("input_ids: ", input_ids.shape, input_ids.dtype)
        # print("attention_mask: ", attention_mask.shape, attention_mask.dtype)

        outputs = self.clm_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # print("outputs['hidden_states'][0]: ", outputs['hidden_states'][0].shape)

        logits = self.score(outputs['hidden_states'][0])

        # print("logits: ", logits.shape, logits.dtype)

        sequence_lengths = -1 # ATM we use only left side padding so it will always work.
        pooled_logits = logits[:, sequence_lengths]

        # print("pooled_logits: ", pooled_logits.shape, pooled_logits.dtype)

        # TODO: Should be put elsewhere
        labels = torch.nn.functional.one_hot(label_id, num_classes=self.num_labels).to(pooled_logits.device, pooled_logits.dtype)

        # print("labels: ", labels.shape, labels.dtype)

        loss_fct = nn.CrossEntropyLoss(self.label_weights)
        loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
mh_model = MultiHeadCLM(clm_model)
mh_model = PeftModel.from_pretrained(mh_model, "results/llama7b_chat_mh_cls")
mh_model = mh_model.to(torch_dtype)

def preprocess(example, max_len):

    turns = [
        {"role": "user", "content": example['text'].strip()},
    ]

    input_ids = tokenizer.apply_chat_template(turns)

    if len(input_ids) > max_len:
        return None

    return {
        'example_id': example['example_id'],
        'text': tokenizer.decode(input_ids),
        'label_id': example['label_id'],
    }

_, test = load_semeval_taskb(return_sets='splits', urls=False, lower=False)

test = [preprocess(example, script_args.max_len) for example in test.to_dict(orient='records')]
test = [example for example in test if example]

def collate_key(batch, key):
  return [ex[key] for ex in batch]


def pad_key(batch, key, pad_value):
  collated = collate_key(batch, key)
  max_len = max([len(ex) for ex in collated])
  return torch.LongTensor([[pad_value] * (max_len - len(ex)) + ex for ex in collated])


def collate(tokenizer):
  def wrapped_collate(batch):
    return {
        'example_id': collate_key(batch, 'example_id'),
        # 'text': collate_key(batch, 'text'),
        'label_id': torch.tensor(collate_key(batch, 'label_id')),
        'input_ids': pad_key(batch, 'input_ids', tokenizer.pad_token_id),
        'attention_mask': pad_key(batch, 'attention_mask', 0),
    }
  return wrapped_collate


def make_loader(data, tokenizer, batch_size, shuffle=True):
    '''
        Create dataset, tokenize examples, pad examples then return a loader.
    '''
    data_set = Dataset.from_list(data).map(lambda x: tokenizer(x['text']))
    return DataLoader(data_set, batch_size=batch_size, collate_fn=collate(tokenizer), shuffle=shuffle)


test_set = Dataset.from_list(test).map(lambda x: tokenizer(x['text']))


loader = make_loader(test, tokenizer, 1, shuffle=False)

mh_model.eval()

act_func = torch.nn.Softmax(dim=1)

results = []

with torch.no_grad():
    for batch in tqdm(loader):
        outputs = mh_model(**batch)
        scores = act_func(outputs.logits)
        results.append(
           {
              'example_id': batch[0]['example_id'],
              'label_id': batch[0]['label_id'].cpu().item(),
              'scores': scores.cpu().tolist(),
              'pred': scores.argmax(dim=1).cpu().item()
           }
        )