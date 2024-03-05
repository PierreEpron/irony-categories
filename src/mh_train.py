from dataclasses import dataclass, field, asdict
import json
from typing import Optional
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import Dataset

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from lightning import LightningModule
from lightning import Trainer

from torch import nn
from torch.utils.data import DataLoader
import torch

from accelerate import init_empty_weights

from src.preprocessing import load_semeval_taskb
from src.utils import write_jsonl, get_hf_token
from src.model import load_clm

@dataclass(frozen=True)
class ScriptArguments:
    # model
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    empty_weight: Optional[bool] = field(default=False, metadata= {"help": "if true init the model with empty_weight"})
    max_len: Optional[int] = field(default=105, metadata={"help":"drop example that have more token than max_len after tokenization"})

    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})

    # training args
    output_dir: Optional[str] = field(default="results/llama7b_chat_mh_cls", metadata={"help": "the output directory"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    num_train_epochs: Optional[int] = field(default=50, metadata={"help": "the number of training epochs"})
    log_every_n_steps: Optional[int] = field(default=1, metadata={"help": "step interval for logging"})
    accelerator: Optional[str] = field(default="gpu", metadata={"help": "accelerator to use"})
    devices: Optional[int] = field(default=1, metadata={"help": "amount of device to use"})
    # early stopping args
    early_stopping_monitor: Optional[str] = field(default="val_loss", metadata={"help": "the value to monitor for validation loop"})
    early_stopping_min_delta: Optional[float] = field(default=0.0, metadata="how much the specified metric must improve to satisfy early stopping conditions.")
    early_stopping_patience: Optional[int] = field(default=5, metadata="stop training when the specified metric worsens for early_stopping_patience evaluation calls")
    early_stopping_mode: Optional[str] = field(default="min", metadata={"help": "the value to monitor for validation loop"})

script_args = ScriptArguments()

output_dir = Path(script_args.output_dir)
if not output_dir.is_dir():
    output_dir.mkdir()

(output_dir / "config.json").write_text(json.dumps(asdict(script_args)))

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


def collate_key(batch, key):
  return [ex[key] for ex in batch]


def pad_key(batch, key, pad_value):
  collated = collate_key(batch, key)
  max_len = max([len(ex) for ex in collated])
  return torch.tensor([[pad_value] * (max_len - len(ex)) + ex for ex in collated])


def collate(tokenizer):
  def wrapped_collate(batch):
    return {
        'example_id': collate_key(batch, 'example_id'),
        'text': collate_key(batch, 'text'),
        'label_id': collate_key(batch, 'label_id'),
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


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


def get_plt_loggers(result_path, name):
    '''
        Shortcut to get plt loggers
    '''

    # tb_logger = TensorBoardLogger(result_path / "tb_logs", name=name)
    csv_logger = CSVLogger(result_path / "cv_logs", name=name)
    return [csv_logger]


class CLMClassifier(nn.Module):
    def __init__(self, context_size=4096, num_labels=4, label_weigths=None) -> None:
        super().__init__()

        self.context_size = context_size
        self.num_labels = num_labels
        self.label_weights = label_weigths

        self.cls_layer = nn.Linear(self.context_size, out_features=self.num_labels)
        self.act_fct = torch.nn.Softmax(dim=1)
        self.loss_fct = nn.CrossEntropyLoss(self.label_weights)

    def forward(self, inputs, targets=None):

        sequence_lengths = -1 # ATM we use only left side padding so it will always work.

        logits = self.cls_layer(logits)
        logits = logits[:, sequence_lengths]
        logits = self.act_fct(logits)

        if targets != None:
          targets = torch.nn.functional.one_hot(targets, num_classes=self.num_labels).float().to(logits.device)
          loss = self.loss_fct(logits, targets)

          return {
            'logits':logits, 'loss':loss
          }

        return {
          'logits':logits
        }


class MultiHeadCLM(LightningModule):
    '''
    '''
    def __init__(self, clm_model, cls_model, learning_rate, device='cuda'):
        super().__init__()
        self.clm_model = clm_model
        self.cls_model = cls_model
        self.learning_rate = learning_rate
        freeze_module(clm_model)


    def forward(self, input_ids, attention_mask, targets):
        clm_outputs = self.clm_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_outputs = self.cls_model(clm_outputs['hidden_states'][0], targets)
        return cls_outputs

    def training_step(self, batch, batch_idx):
        print('train', batch['input_ids'], batch['attention_mask'], batch['label_id'])
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['label_id'])
        loss = outputs['loss']
        self.log("train_loss", loss, batch_size=1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print('val0')
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['label_id'])
        print('val1')
        val_loss = outputs['loss']
        print('val2')
        self.log("val_loss", val_loss, batch_size=1, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        print('test', batch['input_ids'], batch['attention_mask'], batch['label_id'])
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])
        return {
            'id_original':batch['id_original'][0],
            'text':batch['text'][0],
            'gold':batch['label_id'].item(),
            'pred':outputs['logits'].argmax().item(),
            'scores':outputs['logits'].tolist()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left", token=get_hf_token())
tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
tokenizer.use_default_system_prompt = False

clm_model = load_clm(script_args.model_name, empty_weight=script_args.empty_weight)
clm_model.config.pad_token_id = tokenizer.pad_token_id

print("loaded")

train, test = load_semeval_taskb(return_sets='splits', urls=False, lower=False)

train = [preprocess(example, script_args.max_len) for example in train.to_dict(orient='records')]
train = [example for example in train if example]

test = [preprocess(example, script_args.max_len) for example in test.to_dict(orient='records')]
test = [example for example in test if example]
print(len(train), len(test))

train_set = Dataset.from_list(train[len(test):])
val_set = Dataset.from_list(train[:len(test)])
test_set = Dataset.from_list(test)

print(len(train_set), len(val_set), len(test_set))

train_loader = make_loader(train_set, tokenizer, script_args.batch_size, shuffle=True)
val_loader = make_loader(val_set, tokenizer, script_args.batch_size, shuffle=True)
test_loader = make_loader(test_set, tokenizer, 1)


model = MultiHeadCLM(clm_model, CLMClassifier(), script_args.learning_rate)

early_stopping = EarlyStopping(
    monitor=script_args.early_stopping_monitor, 
    min_delta=script_args.early_stopping_min_delta, 
    patience=script_args.early_stopping_patience,
    mode=script_args.early_stopping_mode
)

results = []

trainer = Trainer(
    default_root_dir=str(output_dir),
    max_epochs=script_args.num_train_epochs,
    log_every_n_steps=script_args.log_every_n_steps,
    # Rework path
    logger=get_plt_loggers(str(output_dir), str(output_dir).split('/')[-1]),
    callbacks=[early_stopping],
    accelerator=script_args.accelerator, 
    devices=script_args.devices
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

results.extend(trainer.predict(model, test_loader, ckpt_path='best'))

write_jsonl(output_dir / 'predictions.jsonl')