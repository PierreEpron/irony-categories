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
class ScriptConfig:
    result_path: str = field(metadata={"help":"The path used to store results"})
    model_path: Optional[str] = field(default="dummy_ff_0_1E-05_1E-05/dummy_ff_0_1E-05_1E-05_0", metadata={"help": "Pretrain model name for huggingface."})

parser = HfArgumentParser([M.PretrainedLLMConfig, P.DataConfig, ScriptConfig])
llm_config, data_config, script_config = parser.parse_args_into_dataclasses()

if data_config.dataset not in P.MANAGER_CLASS_MAP:
    raise AttributeError(f"`training_config.dataset` should be equal to ['semeval', 'goemotions'] not to {data_config.dataset}")

trainer = L.Trainer(
    log_every_n_steps=1
)

model = M.LLMClassifier.load(script_config.model_path)

tokenizer = AutoTokenizer.from_pretrained(model.llm_config.model_name)
tokenizer.use_default_system_prompt = False

data_manager = P.MANAGER_CLASS_MAP[data_config.dataset](tokenizer, data_config)
train_loader, val_loader, test_loader = data_manager.get_data_loaders()

predictions = []

for batch in test_loader:

    outputs = model.llm_model.forward(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        output_hidden_states=True,
        return_dict=True,
    )

    logits = model.clf_model.forward_all_token(outputs)
    _scores, _pred = M.single_class_inference.inference(logits)

    for example_id, label_id, previous_text, text, labels, scores, pred in zip(
        batch['example_id'],
        batch['label_id'],
        batch['previous_text'],
        batch['text'],
        batch['labels'],
        _scores,
        _pred
    ):
        predictions.append({
            'example_id':example_id,
            'label_id':label_id,
            'previous_text':previous_text,
            'text':text,
            'labels':labels.cpu().tolist(),
            'scores':scores.cpu().tolist(),
            'pred':pred.cpu().tolist()
        })
        


write_jsonl(Path(script_config.result_path) / "predictions.jsonl", predictions)
