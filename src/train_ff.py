from transformers import HfArgumentParser
from src.training import run
from src import model as M

if __name__ == "__main__":

    ##### Parse args and instantiate configs #####

    parser = HfArgumentParser([M.PretrainedLLMConfig, M.PeftConfig, M.FFClassifierConfig, M.TrainingConfig])
    llm_config, peft_config, clf_config, training_config = parser.parse_args_into_dataclasses()
    run(llm_config, peft_config, clf_config, training_config)