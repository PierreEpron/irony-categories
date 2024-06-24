from transformers import HfArgumentParser
from src.training import run

from src import model as M
from src import preprocessing as P
import warnings


if __name__ == "__main__":

    ##### Parse args and instantiate configs #####

    parser = HfArgumentParser([M.PretrainedLLMConfig, M.PeftConfig, M.FFClassifierConfig, M.TrainingConfig, P.DataConfig])
    llm_config, peft_config, clf_config, training_config, data_config = parser.parse_args_into_dataclasses()

    if data_config.num_labels != clf_config.num_logits:
        warnings.warn("Warning: `data_config.num_labels` ({data_config.num_labels}) is not equal to `clf_config.num_logits` ({clf_config.num_logits}). It could be a mistake !")

    run(llm_config, peft_config, clf_config, training_config, data_config)