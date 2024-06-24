from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor
from transformers import AutoTokenizer
import lightning as L

from src import model as M
from src import preprocessing as P

from src.tools.split_data import get_split
from src.utils import get_hf_token, write_jsonl


def train_model(llm_config, peft_config, clf_config, training_config, data_config):

    ##### Load tokenizer and model #####

    tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name)
    tokenizer.use_default_system_prompt = False

    model = M.LLMClassifier(
        llm_config, 
        clf_config=clf_config, 
        peft_config=peft_config, 
        training_config=training_config, 
        data_config=data_config,
        hf_token=get_hf_token()
    )

    if llm_config.eos_as_pad:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.llm_model.config.pad_token_id = tokenizer.eos_token_id


    ##### Load and preprocess data #####

    if data_config.dataset not in P.MANAGER_CLASS_MAP:
        raise AttributeError(f"`training_config.dataset` should be equal to ['semeval', 'goemotions'] not to {training_config.dataset}")

    data_manager = P.MANAGER_CLASS_MAP[data_config.dataset](tokenizer, data_config)
    train_loader, val_loader, test_loader = data_manager.get_data_loaders()

    ##### Train model #####

    weighted_loss = M.WeightedLossCallback()
    lr_scheduler = M.LRSchedulerCallback()
    lr_monitor = LearningRateMonitor(logging_interval=None)
    ckpt_callback = M.CkptCallback(ckpt_path=training_config.result_path)

    trainer = L.Trainer(
        max_epochs=training_config.max_epochs,
        gradient_clip_val=training_config.gradient_clip_val if training_config.gradient_clip_val != 0 else None,
        enable_checkpointing=False,
        logger=M.get_plt_loggers(training_config.result_path),
        callbacks=[weighted_loss, lr_scheduler, lr_monitor, ckpt_callback],
        log_every_n_steps=1
    )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )


    ##### Evaluate model #####

    model = M.LLMClassifier.load(training_config.result_path)
    predictions = trainer.predict(
        model=model, 
        dataloaders=test_loader, 
        return_predictions=True
    )
    write_jsonl(Path(training_config.result_path) / "predictions.jsonl", predictions)

def run(llm_config, peft_config, clf_config, training_config, data_config):
    
    if data_config.current_split == -1:

        base_path = Path(training_config.result_path)

        if not base_path.is_dir():
            base_path.mkdir()

        for i in range(5):  
            data_config.current_split = i
            training_config.result_path = str(base_path / f"{base_path.parts[-1]}_{i}")  
            train_model(llm_config, peft_config, clf_config, training_config, data_config)

    else:
        train_model(llm_config, peft_config, clf_config, training_config, data_config)