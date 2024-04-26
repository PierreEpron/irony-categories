from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FFClassifierConfig:
    input_size: Optional[int] = field(default=64, metadata={"help":"Size of input. Should be equal to the hidden_states size of the LLM used as input."})
    num_labels: Optional[int] = field(default=4, metadata={"help":"Number of labels. Used as output size of logits."})
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

@dataclass
class PeftConfig:

    lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapter"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapter"})
    target_modules: Optional[list] = field(default_factory=lambda:["q_proj", "v_proj"], metadata={"help": "layers to plug LoRA adapter."})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help":"use or not dropout for the LoRA adapter"})
    lora_bias: Optional[str] = field(default="none", metadata={"help":"use or not bias for the LoRA adapter"})
    inference_mode: Optional[bool] = field(default=False, metadata={"help": "Is the adapter used for inference or not"})

class FFClassifer(torch.nn.Module):

    def __init__(
        self,
        config
    ): 
        super().__init__()
        self.config = config
        self.output_layer = torch.nn.Linear(self.config.input_size, self.config.num_labels)

    def forward(self, inputs):

        logits = self.output_layer(inputs['hidden_states'][self.config.hidden_states_idx])
        pooled_logits = logits[:, self.config.cls_token_idx]
        
        return pooled_logits
    
    def save(self, path):
        
        path = Path(path) if isinstance(path, str) else path
        if not path.is_dir():
            path.mkdir()

        (path / "clf_config.json").write_text(json.dumps(self.config.__dict__), encoding='utf-8')
        torch.save(self.state_dict(), path / "clf_model.bin")

    @classmethod
    def load(cls, path):
        path = Path(path) if isinstance(path, str) else path

        config = FFClassifierConfig(**json.loads((path / "clf_config.json").read_text(encoding='utf-8')))

        model = cls(config)
        model.load_state_dict(torch.load(path / "clf_model.bin"))

        return config, model
    
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

        self.load_llm(llm_config, device_map=device_map, torch_dtype=torch_dtype, hf_token=hf_token)
        self.load_adapter(peft_config, peft_model_name)
        self.load_classifier(clf_config, clf_model_name)
        self.training_config = training_config

        # TODO: Better way to handle
        self.clf_model.to(dtype=torch_dtype)

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
        if config.load_in_4bit or config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=config.load_in_8bit, 
                load_in_4bit=config.load_in_4bit
            )

        self.llm_model = AutoModel.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token
        )

    def load_adapter(self, peft_config = None, peft_model_name = None):
        assert not peft_config or not peft_model_name, f"Both `peft_config` and `peft_model_name` are not None/False."

        if peft_config:
            self.peft_config = peft_config
            self.llm_model = get_peft_model(self.llm_model, LoraConfig(
                r=self.peft_config.lora_alpha,
                lora_alpha=self.peft_config.lora_r,
                target_modules=self.peft_config.target_modules,
                lora_dropout=self.peft_config.lora_dropout,
                bias=self.peft_config.lora_bias,
                inference_mode=self.peft_config.inference_mode
            ))

        elif peft_model_name:
            path = Path(peft_model_name) if isinstance(peft_model_name, str) else peft_model_name
            path = path / "peft_config.json"

            if path.is_file():
                self.peft_config = PeftConfig(**json.loads(path.read_text(encoding='utf-8')))
                self.llm_model = PeftModel.from_pretrained(self.llm_model, peft_model_name, is_trainable=not self.peft_config.inference_mode)

    def load_classifier(self, clf_config=None, clf_model_name=None):
        assert not clf_config or not clf_model_name, f"Both `clf_config` and `clf_model_name` are not None/False."
        
        if clf_config:
            self.clf_config = clf_config
            self.clf_model = FFClassifer(clf_config)
        elif clf_model_name:
            self.config, self.clf_model = FFClassifer.load(clf_model_name)
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
        
        loss = torch.functional.F.cross_entropy(logits, batch['labels'].to(logits.dtype))
        self.log("train_loss", loss)

        # TODO: argmax on labels should be avoided. Maybe move label onehot encoding from dataloader to here.
        self.train_outputs.extend(logits.argmax(dim=-1).cpu())
        self.train_targets.extend(batch['labels'].argmax(dim=-1).cpu())

        return loss
    
    def on_train_epoch_end(self):
        
        self.log("train_mcc", matthews_corrcoef(self.train_outputs, self.train_targets))
        self.train_outputs.clear()
        self.train_targets.clear()
    

    def validation_step(self, batch, batch_idx):

        logits = self.forward(batch, batch_idx)

        loss = torch.functional.F.cross_entropy(logits, batch['labels'].to(logits.dtype))
        self.log("val_loss", loss)

        # TODO: argmax on labels should be avoided. Maybe move label onehot encoding from dataloader to here.
        self.val_outputs.extend(logits.argmax(dim=-1).cpu())
        self.val_targets.extend(batch['labels'].argmax(dim=-1).cpu())

    def on_validation_epoch_end(self):
        print('LLMClassifier.on_validation_epoch_end')

        mcc = matthews_corrcoef(self.val_outputs, self.val_targets)

        self.log("val_mcc", mcc)
        self.val_outputs.clear()
        self.val_targets.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params":self.llm_model.parameters(), "lr":self.training_config.lr_peft},
            {"params":self.clf_model.parameters(), "lr":self.training_config.lr_clf}
        ])
        return optimizer