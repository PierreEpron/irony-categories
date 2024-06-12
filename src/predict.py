from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json


from transformers import AutoTokenizer, HfArgumentParser
import lightning as L

from src.utils import get_hf_token, write_jsonl
from src import preprocessing as P
from src import model as M

@dataclass
class InferenceConfig:
    result_path: str = field(metadata={"help":"The path used to store results"})
    dataset: Optional[str] = field(default="semeval", metadata={"help":"The dataset used to train the model."})
    model_name: Optional[str] = field(default="dummy_ff_0_1E-05_1E-05/dummy_ff_0_1E-05_1E-05_0", metadata={"help": "Pretrain model name for huggingface."})

parser = HfArgumentParser([InferenceConfig])
inference_config = parser.parse_args_into_dataclasses()[0]


if inference_config.dataset == 'semeval':
    data = P.SemEval.load_data(return_sets="full", urls=False, lower=False).to_dict(orient='records')

elif inference_config.dataset == 'goemotions':
    data = P.GoEmotions.load_data(return_sets='full')

else:
    raise AttributeError(f"`training_config.dataset` should be equal to ['semeval', 'goemotions'] not to {inference_config.dataset}")


model_name = f"results/{inference_config.model_name}"

model = M.LLMClassifier.load(model_name)

tokenizer = AutoTokenizer.from_pretrained(model.llm_config.model_name)
tokenizer.use_default_system_prompt = False

turns = [{"role": "user", "content": "{text}"}]

data_set = P.make_dataset(data, tokenizer, turns, num_classes=model.clf_config.num_labels, max_len=4096)

print(len(data_set))

data_loader = P.make_loader(data_set, tokenizer, 1, True, False)

trainer = L.Trainer()

predictions = trainer.predict(
    model=model, 
    dataloaders=data_loader, 
    return_predictions=True
)

write_jsonl(Path(inference_config.result_path), predictions)