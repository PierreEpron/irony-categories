from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor
from transformers import AutoTokenizer
import lightning as L

from src import model as M
from src import preprocessing as P

from src.tools.split_data import get_split
from src.utils import get_hf_token, write_jsonl


def train_model(llm_config, peft_config, clf_config, training_config):

    ##### Load tokenizer and model #####

    tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name)
    tokenizer.use_default_system_prompt = False

    model = M.LLMClassifier(
        llm_config, 
        clf_config=clf_config, 
        peft_config=peft_config, 
        training_config=training_config, 
        hf_token=get_hf_token()
    )

    if llm_config.eos_as_pad:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.llm_model.config.pad_token_id = tokenizer.eos_token_id


    ##### Load and preprocess data #####

    if training_config.dataset == 'semeval':
        train, test = P.SemEval.load_data(return_sets="splits", urls=False, lower=False)
        train, val = get_split(training_config.current_split, training_config.split_path, train)

    elif training_config.dataset == 'goemotions':
        train, val, test = P.GoEmotions.load_data(return_sets='splits')

    else:
        raise AttributeError(f"`training_config.dataset` should be equal to ['semeval', 'goemotions'] not to {training_config.dataset}")

    turns = [{"role": "user", "content": "{text}"}]

    train_set = P.make_dataset(train, tokenizer, turns, max_len=training_config.max_len)
    val_set = P.make_dataset(val, tokenizer, turns, max_len=training_config.max_len)
    test_set = P.make_dataset(test, tokenizer, turns, max_len=training_config.max_len)

    # print(f"{len(train_set)}/{len(train)}, {len(val_set)}/{len(val)}, {len(test_set)}/{len(test)}")

    train_loader = P.make_loader(train_set, tokenizer, training_config.train_batch_size, False, True)
    val_loader = P.make_loader(val_set, tokenizer, training_config.val_batch_size, False, False)
    test_loader = P.make_loader(test_set, tokenizer, training_config.test_batch_size, True, False)

    # print(next(iter(train_loader)).keys())
    # print(next(iter(test_loader)).keys())
    # print(next(iter(train_loader))['input_ids'].shape)

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

def run(llm_config, peft_config, clf_config, training_config):
    
    if training_config.current_split == -1:

        base_path = Path(training_config.result_path)

        if not base_path.is_dir():
            base_path.mkdir()

        for i in range(5):  
            training_config.current_split = i
            training_config.result_path = str(base_path / f"{base_path.parts[-1]}_{i}")  
            train_model(llm_config, peft_config, clf_config, training_config)

    else:
        train_model(llm_config, peft_config, clf_config, training_config)