from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.utilities.types import LRSchedulerConfig
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef
import lightning as L
import numpy as np
import torch


@dataclass
class FFClassifierConfig:
    input_size: Optional[int] = field(default=4096, metadata={"help":"Size of input. Should be equal to the hidden_states size of the LLM used as input."})
    num_labels: Optional[int] = field(default=4, metadata={"help":"Number of labels. Used as output size of logits."})
    hidden_states_idx: Optional[int] = field(default=-1, metadata={"help":"Index of hidden_states block to use as input for classification."})
    cls_token_idx: Optional[int] = field(default=-1, metadata={"help":"Index of hidden_states token to use as input for classification."})

@dataclass
class MLPClassifierConfig:
    input_size: Optional[int] = field(default=4096, metadata={"help":"Size of input. Should be equal to the hidden_states size of the LLM used as input."})
    n_layers: Optional[int] = field(default=1, metadata={"help":"Number of hidden layers before output layer. Input is divided by 2 for each layer."})
    num_labels: Optional[int] = field(default=4, metadata={"help":"Number of labels. Used as output size of logits."})
    dropout_rate: Optional[float] = field(default=0.5, metadata={"help":"Dropout rate to use for each dropout layer of the classifier. A dropout layer is added after the activation function of each hidden layer."})
    hidden_states_idx: Optional[int] = field(default=-1, metadata={"help":"Index of hidden_states block to use as input for classification."})
    cls_token_idx: Optional[int] = field(default=-1, metadata={"help":"Index of hidden_states token to use as input for classification."})

@dataclass
class PretrainedLLMConfig:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "pretrain model name for huggingface."})
    # tokenizer args
    padding_side: Optional[str] = field(default="left", metadata={"help": "padding side of the tokenizer."})
    eos_as_pad: Optional[bool] = field(default=True, metadata={"help": "use eos token as pad token."})
    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision."})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision."})
    freeze_layers: Optional[bool] = field(default=True, metadata={"help": "Either to freeze or not layers of the LLM"})

@dataclass
class PeftConfig:

    use_lora: Optional[bool] = field(default=True, metadata={"help": "Either to add or not a LoRA adapter to the LLM. If False all the other parameters are useless."})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapter"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapter"})
    target_modules: Optional[str] = field(default="q_proj,v_proj", metadata={"help": "layers to plug LoRA adapter. Splitted by comma."})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help":"use or not dropout for the LoRA adapter"})
    lora_bias: Optional[str] = field(default="none", metadata={"help":"use or not bias for the LoRA adapter"})
    inference_mode: Optional[bool] = field(default=False, metadata={"help": "Is the adapter used for inference or not"})

@dataclass
class TrainingConfig:

    result_path: str = field(metadata={"help": "The path used to store results"})
    current_split: Optional[int] = field(default=-1, metadata={"help": "The cross validation split to use (between 0 and 4). -1 is used to run on all splits"})
    split_path: Optional[str] = field(default="data/sem_eval/splits.jsonl", metadata={"help":"The jsonl file path containing the cross validation split indices"})
    clf_class: Optional[str] = field(default="ff", metadata={"help":"Keyword used to recover correct classifier class: [ff, mlp]"})
    lr_peft: Optional[float] = field(default=1.5e-4, metadata={"help":"Learning rate for peft adapater on LLM"})
    lr_clf: Optional[float] = field(default=1e-3, metadata={"help":"Learning rate for classifier"})
    weighted_loss: Optional[str] = field(default="no", metadata={"help":"Method to use for weight loss"})
    gradient_clip_val: Optional[float] = field(default=0, metadata={"help":"Clip gradients' global norm to <=`gradient_clip_val` using torch.nn.utils.clip_grad_norm_()"})
    max_epochs: Optional[int] = field(default=10, metadata={"help":"Maximum number of epochs"})
    lr_scheduler: Optional[str] = field(default="no", metadata={"help":"Which kind of lr scheduler to use. If `no` don't use one."})
    num_warmup_epoch: Optional[int] = field(default=1, metadata={"help":"If `lr_scheduler` is not `no`. Nb of epochs to use as warmup."})
    train_batch_size: Optional[int] = field(default=16, metadata={"help":"Size of a train batch"})
    val_batch_size: Optional[int] = field(default=16, metadata={"help":"Size of a validation batch"})
    test_batch_size: Optional[int] = field(default=1, metadata={"help":"Size of a test batch"})
    max_len: Optional[int] = field(default=105, metadata={"help":"Maximum length of a tokenized example. If greater than this length, drop the example."})


class BaseClassifer(torch.nn.Module):
    config_class = None

    def __init__(self, config):
        super().__init__()
        self.config = config

    def save(self, path):
        
        path = Path(path) if isinstance(path, str) else path
        if not path.is_dir():
            path.mkdir()

        (path / "clf_config.json").write_text(json.dumps(self.config.__dict__), encoding='utf-8')
        torch.save(self.state_dict(), path / "clf_model.bin")

    @classmethod
    def load(cls, path):
        path = Path(path) if isinstance(path, str) else path

        config = cls.config_class(**json.loads((path / "clf_config.json").read_text(encoding='utf-8')))

        model = cls(config)
        model.load_state_dict(torch.load(path / "clf_model.bin"))

        return config, model


class FFClassifer(BaseClassifer):
    config_class = FFClassifierConfig

    def __init__(self, config): 
        super().__init__(config)
        
        self.output_layer = torch.nn.Linear(self.config.input_size, self.config.num_labels)

    def forward(self, inputs):

        logits = self.output_layer(inputs['hidden_states'][self.config.hidden_states_idx])
        pooled_logits = logits[:, self.config.cls_token_idx]
        
        return pooled_logits


class MLPBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout_rate, act_func):
        super().__init__()

        self.layer = torch.nn.Linear(input_size, output_size)
        self.act_func = act_func
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        return self.dropout(self.act_func(self.layer(inputs)))

class MLPClassifier(BaseClassifer):
    config_class = MLPClassifierConfig

    def __init__(self, config): 
        super().__init__(config)

        self.hidden_layers = []

        input_size = self.config.input_size
        for _ in range(self.config.n_layers):
            output_size = self.config.input_size // 2
            self.hidden_layers.append(MLPBlock(input_size, output_size, self.config.dropout_rate, torch.nn.ReLU()))
            input_size = output_size

        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)
        self.output_layer = torch.nn.Linear(input_size, self.config.num_labels)

    def forward(self, inputs):

        logits = inputs['hidden_states'][self.config.hidden_states_idx]

        for layer in self.hidden_layers:
            logits = layer(logits)

        logits = self.output_layer(logits)
        pooled_logits = logits[:, self.config.cls_token_idx]
        
        return pooled_logits


CLF_MODULE_MAP = {
    "ff":FFClassifer,
    "mlp":MLPClassifier
}

class LLMClassifier(L.LightningModule):
    def __init__(
        self, 
        llm_config,
        peft_config = None,
        peft_model_name = None, 
        clf_config = None, 
        clf_model_name = None,
        training_config = None,
        device_map={"":0}, 
        torch_dtype=torch.bfloat16, 
        hf_token=None
    ):
        
        super().__init__()

        self.training_config = training_config

        self.load_llm(llm_config, device_map=device_map, torch_dtype=torch_dtype, hf_token=hf_token)
        self.load_adapter(peft_config, peft_model_name)
        self.load_classifier(clf_config, clf_model_name)

        # TODO: Better way to handle
        self.clf_model.to(dtype=torch_dtype)
        self.torch_dtype = torch_dtype
        self.label_weigths = None

        # Use to compute MCC at the end of each train/val epoch.
        self.train_outputs = []
        self.train_targets = []
        self.val_outputs = []
        self.val_targets = []

    def load_llm(self, llm_config, device_map={"":0}, torch_dtype=torch.bfloat16, hf_token=None):
        
        self.llm_config = llm_config
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        quantization_config = None
        if self.llm_config.load_in_4bit or self.llm_config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.llm_config.load_in_8bit, 
                load_in_4bit=self.llm_config.load_in_4bit
            )

        self.llm_model = AutoModel.from_pretrained(
            self.llm_config.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token
        )

        if self.llm_config.freeze_layers:
            for param in self.llm_model.parameters():
                param.requires_grad = False
        else:
            for param in self.llm_model.parameters():
                param.requires_grad = True

    def load_adapter(self, peft_config = None, peft_model_name = None):
        assert not peft_config or not peft_model_name, f"Both `peft_config` and `peft_model_name` are not None/False."

        if peft_config:
            self.peft_config = peft_config
            if self.peft_config.use_lora:
                self.llm_model = get_peft_model(self.llm_model, LoraConfig(
                    r=self.peft_config.lora_r,
                    lora_alpha=self.peft_config.lora_alpha,
                    target_modules=self.peft_config.target_modules.split(','),
                    lora_dropout=self.peft_config.lora_dropout,
                    bias=self.peft_config.lora_bias,
                    inference_mode=self.peft_config.inference_mode
                ))

        elif peft_model_name:
            path = Path(peft_model_name) if isinstance(peft_model_name, str) else peft_model_name
            path = path / "peft_config.json"

            if path.is_file():
                self.peft_config = PeftConfig(**json.loads(path.read_text(encoding='utf-8')))
                if self.peft_config.use_lora:                
                    self.llm_model = PeftModel.from_pretrained(self.llm_model, peft_model_name, is_trainable=not self.peft_config.inference_mode)

    def load_classifier(self, clf_config=None, clf_model_name=None):
        assert not clf_config or not clf_model_name, f"Both `clf_config` and `clf_model_name` are not None/False."
        
        clf_class = CLF_MODULE_MAP[self.training_config.clf_class]

        if clf_config:
            self.clf_config = clf_config
            self.clf_model = clf_class(clf_config)
        elif clf_model_name:
            self.config, self.clf_model = clf_class.load(clf_model_name)
        else:
            raise AttributeError("Neither `clf_config` nor `clf_model_name` are valid.")

    def save(self, path):
        path = Path(path) if isinstance(path, str) else path
        if not path.is_dir():
            path.mkdir()

        (path / "llm_config.json").write_text(json.dumps(self.llm_config.__dict__), encoding='utf-8')
        (path / "peft_config.json").write_text(json.dumps(self.peft_config.__dict__), encoding='utf-8')
        # TODO: Should be optional 
        (path / "training_config.json").write_text(json.dumps(self.training_config.__dict__), encoding='utf-8')
        if self.peft_config.use_lora:
            self.llm_model.save_pretrained(path)
        self.clf_model.save(path)

    @classmethod
    def load(cls, path):
        path = Path(path) if isinstance(path, str) else path
        llm_config = PretrainedLLMConfig(**json.loads((path / "llm_config.json").read_text(encoding='utf-8'))) 
        # TODO: Should be optional 
        training_config = TrainingConfig(**json.loads((path / "training_config.json").read_text(encoding='utf-8')))  
        return cls(llm_config, clf_model_name=path, peft_model_name=path, training_config=training_config)


    def forward(self, batch, batch_idx):
        outputs = self.llm_model.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True,
            return_dict=True,
        )
        logits = self.clf_model(outputs)
        return logits

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch, batch_idx)
        
        loss = torch.functional.F.cross_entropy(logits, batch['labels'].to(logits.dtype), weight=self.label_weigths)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        # TODO: argmax on labels should be avoided. Maybe move label onehot encoding from dataloader to here.
        self.train_outputs.extend(logits.argmax(dim=-1).cpu())
        self.train_targets.extend(batch['labels'].argmax(dim=-1).cpu())

        return loss
    
    def on_train_epoch_end(self):
        
        self.log("train_mcc", matthews_corrcoef(self.train_outputs, self.train_targets), on_step=False, on_epoch=True)
        self.train_outputs.clear()
        self.train_targets.clear()
    

    def validation_step(self, batch, batch_idx):

        logits = self.forward(batch, batch_idx)

        loss = torch.functional.F.cross_entropy(logits, batch['labels'].to(logits.dtype), weight=self.label_weigths)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        # TODO: argmax on labels should be avoided. Maybe move label onehot encoding from dataloader to here.
        self.val_outputs.extend(logits.argmax(dim=-1).cpu())
        self.val_targets.extend(batch['labels'].argmax(dim=-1).cpu())

    def on_validation_epoch_end(self):

        self.log("val_mcc", matthews_corrcoef(self.val_outputs, self.val_targets), on_step=False, on_epoch=True)
        self.val_outputs.clear()
        self.val_targets.clear()


    def predict_step(self, batch, batch_idx):
        logits = self.forward(batch, batch_idx)
        scores = torch.nn.functional.softmax(logits, dim=-1)
        pred = scores.argmax(dim=-1)

        predictions = []

        for example_id, label_id, text, scores, pred in zip(
            batch['example_id'],
            batch['label_id'],
            batch['text'],
            scores.cpu().tolist(),
            pred.cpu()
        ):
            predictions.append({
                'example_id':example_id,
                'label_id':label_id,
                'text':text,
                'scores':scores,
                'pred':pred.item()
            })
        return predictions

    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW([
            {"params":self.llm_model.parameters(), "lr":self.training_config.lr_peft},
            {"params":self.clf_model.parameters(), "lr":self.training_config.lr_clf}
        ])

        return optimizer


def linear_schedule_with_warmup(num_warmup_steps, num_training_steps):
    def _linear_schedule_with_warmup(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return _linear_schedule_with_warmup


def get_plt_loggers(path):
    '''
        Shortcut to get plt loggers
    '''
    path = Path(path) if isinstance(path, str) else path
    if not path.is_dir():
        path.mkdir()

    tb_logger = TensorBoardLogger(path / "tb_logs", name=path.stem)
    csv_logger = CSVLogger(path / "cv_logs", name=path.stem)
    return [csv_logger, tb_logger]


class CkptCallback(L.Callback):
    def __init__(self, ckpt_path, metric_name="val_mcc", best_model_metric=-2):
        self.ckpt_path = ckpt_path
        self.metric_name = metric_name
        self.best_model_metric = best_model_metric

    # TODO: I used training epoch to be sure that this call is made after validation_epoch_end of pl_module
    def on_train_epoch_end(self, trainer, pl_module):
        current_model_metric = trainer.logged_metrics[self.metric_name]
        if self.best_model_metric < current_model_metric:
            self.best_model_metric = current_model_metric
            pl_module.save(self.ckpt_path)
    

class WeightedLossCallback(L.Callback):

    def on_train_start(self, trainer, pl_module):
        
        if pl_module.training_config.weighted_loss == "no":
            pl_module.label_weigths = None
        elif pl_module.training_config.weighted_loss == "balanced":
            label_ids = trainer.train_dataloader.dataset['label_id']
            label_weights = compute_class_weight(
                class_weight="balanced", 
                classes=np.unique(label_ids), 
                y=label_ids
            )
            label_weights = torch.tensor(label_weights).to(dtype=pl_module.torch_dtype, device=pl_module.device)
            pl_module.label_weights = label_weights


class LRSchedulerCallback(L.Callback):
    
    def on_train_start(self, trainer, pl_module):

        if pl_module.training_config.lr_scheduler == "linear":
            
            config = pl_module.training_config
            num_epoch_steps = np.ceil(len(trainer.train_dataloader.dataset) / config.train_batch_size)
            num_warmup_steps = num_epoch_steps * config.num_warmup_epoch
            num_training_steps = num_epoch_steps * config.max_epochs

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                pl_module.optimizers(), 
                lr_lambda=linear_schedule_with_warmup(num_warmup_steps, num_training_steps)
            )

            lr_scheduler_config = LRSchedulerConfig(
                scheduler=lr_scheduler,
                interval="step"
            )

            trainer.strategy.lr_scheduler_configs = [lr_scheduler_config]