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