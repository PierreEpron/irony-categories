from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:

    current_split: int = field(metadata={"help": "The cross validation split to use (between 0 and 4)"})
    result_path: str = field(metadata={"help": "The path used to store results"})

    split_path: Optional[str] = field(default="data/sem_eval/splits.jsonl", metadata={"help":"The jsonl file path containing the cross validation split indices"})

    lr_peft: Optional[float] = field(default=1e-3, metadata={"help":"Learning rate for peft adapater on LLM"})
    lr_clf: Optional[float] = field(default=0.1, metadata={"help":"Learning rate for classifier"})
    max_epochs: Optional[int] = field(default=10, metadata={"help":"Maximum number of epochs"})
    train_batch_size: Optional[int] = field(default=4, metadata={"help":"Size of a train batch"})
    val_batch_size: Optional[int] = field(default=4, metadata={"help":"Size of a validation batch"})
    max_len: Optional[int] = field(default=105, metadata={"help":"Maximum length of a tokenized example. If greater than this length, drop the example."})