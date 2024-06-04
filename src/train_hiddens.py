from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from transformers import HfArgumentParser
from src.training import run
from src import model as M

from tqdm import tqdm
import shutil

import transformers
transformers.logging.set_verbosity_error()

import datasets
datasets.utils.logging.set_verbosity_error()
datasets.disable_progress_bars()

from lightning.pytorch.utilities.warnings import PossibleUserWarning
import warnings
warnings.filterwarnings("ignore", category=PossibleUserWarning)

import logging
log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)

# 7b = 33

@dataclass
class ScriptConfig:
    hidden_states_size: Optional[int] = field(default=33, metadata={"help":"Size of hidden_states tuple. Will be used to train a model on each index of the tuples"})

if __name__ == "__main__":

    ##### Parse args and instantiate configs #####

    parser = HfArgumentParser([ScriptConfig, M.PretrainedLLMConfig, M.PeftConfig, M.FFClassifierConfig, M.TrainingConfig])
    script_config, llm_config, peft_config, clf_config, training_config = parser.parse_args_into_dataclasses()

    result_path = Path(training_config.result_path)
    hidden_path = Path(f'{result_path}_hidden')

    if not hidden_path.is_dir():
        hidden_path.mkdir()

    for i in tqdm(list(range(script_config.hidden_states_size)), desc="Main Loop"):
        
        # reset training_config
        training_config.result_path = str(result_path)
        training_config.current_split = -1

        if (hidden_path / f'{result_path.parts[-1]}_0_h{i}.jsonl').is_file():
            print('Skipped because existing:', hidden_path / f'{result_path.parts[-1]}_0_h{i}.jsonl')
            continue

        clf_config.hidden_states_idx = i
        run(llm_config, peft_config, clf_config, training_config)
        
        for path in result_path.glob('*'):
            current_prediction_path = path / "predictions.jsonl"
            new_prediction_path = hidden_path / f'{path.parts[-1]}_h{i}.jsonl'
            shutil.copyfile(current_prediction_path, new_prediction_path)
            shutil.rmtree(path)
