from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json

from torch.utils.data import DataLoader
from datasets import Dataset
import lightning as L
import pandas as pd
import torch
import re

from src.utils import read_jsonl


HASHTAG_IRONY_PATTERN = re.compile(r'#irony', flags=re.I)
HASHTAG_SARCASM_PATTERN = re.compile(r'#sarcasm', flags=re.I)
HASHTAG_NOT_PATTERN = re.compile(r'#(not)', flags=re.I)
URL_PATTERN = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)", flags=re.I)
USER_PATTERN = re.compile(r'@[^\s]+', flags=re.I)
SPACES_PATTERN = re.compile(r'\s+')

LOADER_MAP = {
    ".json": lambda path: json.loads(path.read_text()),
    ".jsonl": lambda path: read_jsonl(path),
    ".tsv": lambda path: pd.read_csv(path, sep='\t').to_dict(orient="records"),
    ".csv": lambda path: pd.read_csv(path).to_dict(orient="records")
}


@dataclass
class DataConfig:

    dataset: Optional[str] = field(default="semeval", metadata={"help":"The dataset used to train the model."})
    train_path: Optional[str] = field(default="data/sem_eval/train_emoji.jsonl", metadata={"help":"."})
    test_path: Optional[str] = field(default="data/sem_eval/test_emoji.jsonl", metadata={"help":"."})
    additional_context_path: Optional[str] = field(default="", metadata={"help":"."})

    contents_path: Optional[str] = field(default="data/prompts/cls/contents/text.txt", metadata={"help":"."})
    turns_path: Optional[str] = field(default="data/prompts/cls/turns/user_text.json", metadata={"help":"."})
    splits_path: Optional[str] = field(default="data/sem_eval/splits.jsonl", metadata={"help":"The jsonl file path containing the cross validation split indices"})
    current_split: Optional[int] = field(default=-1, metadata={"help": "The cross validation split to use (between 0 and 4). -1 is used to run on all splits"})

    num_labels: Optional[int] = field(default=4, metadata={"help":"Number of labels. Used as output size of logits."})
    max_len: Optional[int] = field(default=300, metadata={"help":"Maximum length of a tokenized example. If greater than this length, drop the example."})
    remove_last_token: Optional[bool] = field(default=False, metadata={"help":"Maximum length of a tokenized example. If greater than this length, drop the example."})

    train_batch_size: Optional[int] = field(default=16, metadata={"help":"Size of a train batch"})
    val_batch_size: Optional[int] = field(default=16, metadata={"help":"Size of a validation batch"})
    test_batch_size: Optional[int] = field(default=1, metadata={"help":"Size of a test batch"})

    hashtag_irony: Optional[bool] = field(default=True, metadata={"help":"Remove '#irony' during preprocessing. Not sensitive to case."}) 
    hashtag_sarcasm: Optional[bool] = field(default=True, metadata={"help":"Remove '#sarcasm' during preprocessing. Not sensitive to case."})
    hashtag_not: Optional[bool] = field(default=True, metadata={"help":"Replace '#not' by 'not' during preprocessing. Not sensitive to case."})
    user_mention: Optional[bool] = field(default=True, metadata={"help":"Replace '@[user]' by '@user' during preprocessing. Not sensitive to case."})
    urls: Optional[bool] = field(default=False, metadata={"help":"Remove '[url]' during preprocessing. Not sensitive to case."})
    lower: Optional[bool] = field(default=False, metadata={"help":"Lowercase during preprocessing."})

def protect_double_brackets(
    text,
    double_left_bracket="__DLBRACKET__",
    double_right_bracket="__DRBRACKET__"
):
    return text.replace('{{', double_left_bracket).replace('}}', double_right_bracket)

def unprotect_double_brackets(
    text, 
    double_left_bracket="__DLBRACKET__", 
    double_right_bracket="__DRBRACKET__"
):
    return text.replace(double_left_bracket, '{').replace(double_right_bracket, '}')

def protect_brackets(
    text, 
    left_bracket="__LBRACKET__",
    right_bracket="__RBRACKET__",
):
    return text.replace('{', left_bracket).replace('}', right_bracket)

def unprotect_brackets(
    text, 
    left_bracket="__LBRACKET__", 
    right_bracket="__RBRACKET__"
):
    return text.replace(left_bracket, '{').replace(right_bracket, '}')


def format_closure(text, content):
    previous_text = None
    while previous_text != text:
        text = protect_double_brackets(text)
        previous_text = text
        text = text.format(**content)
    return unprotect_double_brackets(unprotect_brackets(text))

def format_turns(turns, example, contents={}):
    return [{"role":turn["role"], "content":format_closure(turn["content"], example | contents)} for turn in turns]

def tokenize_turns(turns, contents={}, remove_last_token=False):
    def _tokenize_turns(tokenizer, example, **kwargs):
        tokens = tokenizer.apply_chat_template(format_turns(turns, example, contents), **kwargs)
        return tokens[:-1] if remove_last_token else tokens
    return _tokenize_turns

def tokenize_contents(contents):
    def _tokenize_contents(tokenizer, example, **kwargs):
        return tokenizer.encode(format_closure(contents, example), **kwargs)
    return _tokenize_contents


def get_split(split_id, splits, examples, split_keys=["train", "val"]):
    return [list(filter(lambda x: x['example_id'] in splits[split_id][k], examples)) for k in split_keys]


def collate_key(batch, key):
  return [ex[key] for ex in batch]

def pad_key(batch, key, pad_value):
    collated = collate_key(batch, key)
    max_len = max([len(ex) for ex in collated])
    return [[pad_value] * (max_len - len(ex)) + ex for ex in collated]

def collate(tokenizer, extra_columns=False, dtype=torch.long):
    def wrapped_collate(batch):
        collated_batch = {
            'labels': torch.tensor(collate_key(batch, 'labels')).to(dtype=dtype),
            'input_ids': torch.tensor(pad_key(batch, 'input_ids', tokenizer.pad_token_id)).to(dtype=dtype),
            'attention_mask': torch.tensor(pad_key(batch, 'attention_mask', 0)).to(dtype=dtype),
        }

        if extra_columns:

            collated_batch.update({
                'example_id': collate_key(batch, 'example_id'), 
                'label_id': collate_key(batch, 'label_id'), 
                'text': collate_key(batch, 'text')
            })
            
        return collated_batch
        
    return wrapped_collate


class DataManager:

    def __init__(self, tokenizer, data_config):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.data_config = data_config

        # Setup tokenize_text function with turns and contents

        contents_path = Path(self.data_config.contents_path) 
        turns_path = Path(self.data_config.turns_path)

        # Load contents
        
        if contents_path.is_file():
            self.contents = {f"{contents_path.stem}_content":contents_path.read_text()}
        elif contents_path.is_dir():
            self.contents = {f"{path.stem}_content":path.read_text() for path in contents_path.glob('*.txt')}
        else:
            AttributeError(f"contents_path should be an existing file or dir path. {contents_path}")

        # Load turns and setup tokenize_text

        if turns_path.is_file():
            self.turns = json.loads(turns_path.read_text())
            self.tokenize_text = tokenize_turns(self.turns, self.contents, self.data_config.remove_last_token)
        else:
            self.tokenize_text = tokenize_contents(list(self.contents.values())[0])

        # setup clean functions

        self.clean_funcs = []

        if self.data_config.hashtag_irony:
            self.clean_funcs.append(lambda x: HASHTAG_IRONY_PATTERN.sub(r'', x))
        if self.data_config.hashtag_sarcasm:
            self.clean_funcs.append(lambda x: HASHTAG_SARCASM_PATTERN.sub(r'', x))
        if self.data_config.hashtag_not:
            self.clean_funcs.append(lambda x: HASHTAG_NOT_PATTERN.sub(r'\1', x))
        if self.data_config.user_mention:
            self.clean_funcs.append(lambda x: USER_PATTERN.sub(r'@user', x))
        if self.data_config.urls:
            self.clean_funcs.append(lambda x: URL_PATTERN.sub(r'', x))
        if self.data_config.lower:
            self.clean_funcs.append(lambda x: x.lower())

        # Make any multiple space a single space and strip the str
        self.clean_funcs.append(lambda x: SPACES_PATTERN.sub(r' ', x).strip())
        
    def tokenize_example(self, example):
    
        example['input_ids'] = self.tokenize_text(
            self.tokenizer,
            # Temporarily Replace "{", "}" by custom token to avoid format issues.
            {k:protect_brackets(v) if isinstance(v, str) else v for k, v in example.items()}
        )
        example['attention_mask'] = [1] * len(example['input_ids'])

        # One hot labels if they doesn't exist
        if 'labels' not in example:
            example['labels'] = [0] * self.num_class
            example['labels'][example['label_id']] = 1

        return example

    def process_examples(self, examples, max_len=-1):
        
        examples = Dataset.from_list(examples)
        examples = examples.map(self.tokenize_example)

        if max_len > 0:
            examples = examples.filter(lambda x: len(x['input_ids']) < max_len)
            
        return examples
    
    def clean_texts(self, examples, keys=['text']):

        for example in examples:
            for key in keys:
                for func in self.clean_funcs:
                    example[key] = func(example[key])

        return examples

    def get_examples_loader(self, examples, batch_size, extra_columns=False, shuffle=False):

        return DataLoader(
            examples, 
            batch_size=batch_size, 
            collate_fn=collate(self.tokenizer, extra_columns=extra_columns), 
            shuffle=shuffle
        )

    def load_examples(self, path, additional_context, additional_context_keys):
        path = Path(path)
        
        if path.is_file():
            examples = LOADER_MAP[path.suffix](path)
            examples = [example | additional_context[example['example_id']] if example['example_id'] in additional_context else {} for example in examples]
            return self.clean_texts(examples, keys=['text'] + additional_context_keys)
        
        return []

    def load_data(self):

        train_examples, val_examples, test_examples = [], [], []

        # Load additional context
        additional_context_path = Path(self.data_config.additional_context_path)
        additional_context = LOADER_MAP[additional_context_path.suffix](additional_context_path) if additional_context_path.is_file() else {}
        additional_context_keys = list(list(additional_context.values())[0].keys()) if len(additional_context) > 0 else [] 

        # Load examples
        train_examples = self.load_examples(self.data_config.train_path, additional_context, additional_context_keys)
        test_examples = self.load_examples(self.data_config.test_path, additional_context, additional_context_keys)

        # Load splits and split train examples in train/val examples
        splits_path = Path(self.data_config.splits_path)

        if splits_path.is_file():
            splits = LOADER_MAP[splits_path.suffix](splits_path)
            if 0 <= self.data_config.current_split < len(splits): 

                train_examples, val_examples = get_split(
                    self.data_config.current_split, splits, train_examples
                )

        # Return train/val/test examples
        return train_examples, val_examples, test_examples

    def process_data(self):
        
        train_examples, val_examples, test_examples = self.load_data()

        return (
            self.process_examples(train_examples, max_len=self.data_config.max_len),
            self.process_examples(val_examples, max_len=self.data_config.max_len),
            self.process_examples(test_examples)
        )
    
    def get_data_loaders(self):

        train_examples, val_examples, test_examples = self.process_data()
        
        return (
            self.get_examples_loader(train_examples, self.data_config.train_batch_size, shuffle=True),
            self.get_examples_loader(val_examples, self.data_config.val_batch_size),
            self.get_examples_loader(test_examples, self.data_config.test_batch_size, extra_columns=True)
        )


class SemEval(DataManager):
    
    train_path = "data/sem_eval/train_emoji.tsv"
    test_path = "data/sem_eval/test_emoji.tsv"
    
    num_class = 4

    class_labels = ["No Irony", "Irony by Clash", "Situational Irony", "Other Irony"]
    label_2_id = {c:i for i, c in enumerate(class_labels)}
    id_2_label = {i:c for i, c in enumerate(class_labels)}

    # Old way of loading data
    @staticmethod
    def load_raw_data(
            return_sets : str = 'all', 
            hashtag_labels = ' ', 
            hashtag_nots = 'not',
            users='@user', 
            urls = ' ', 
            spaces = ' ', 
            lower = True
        ):
        """ Load and preprocess example for semeval taskb. 

            Parameters
            ----------
            return_sets : str
                ...

            Returns
            -------
            data : tuple
                ...
        """
    
        assert return_sets in ['all', 'full', 'splits'], \
            f"return_sets should be set to 'all', 'full' or 'splits'. Given value {return_sets}"

        # Load tsv files
        train = pd.read_csv(SemEval.train_path, sep='\t', encoding='utf-8')
        test = pd.read_csv(SemEval.test_path, sep='\t', encoding='utf-8')

        # set example ids
        train['example_id'] = train.example_id.apply(lambda x: f"train_{x}")
        test['example_id'] = test.example_id.apply(lambda x: f"test_{x}")

        # If `hashtags` not false, replace #irony #sarcasm by `hashtags` for each example
        if hashtag_labels != False:
            train.text = train.text.apply(lambda x: HASHTAG_SARCASM_PATTERN.sub(hashtag_labels, HASHTAG_IRONY_PATTERN.sub(hashtag_labels, x)))
            test.text = test.text.apply(lambda x: HASHTAG_SARCASM_PATTERN.sub(hashtag_labels, HASHTAG_IRONY_PATTERN.sub(hashtag_labels, x)))

        # If `hashtag_nots` not false, replace #not by `hashtag_nots` for each example
        if hashtag_nots != False:
            train.text = train.text.apply(lambda x: HASHTAG_NOT_PATTERN.sub(hashtag_nots, x))
            test.text = test.text.apply(lambda x: HASHTAG_NOT_PATTERN.sub(hashtag_nots, x))

        # If `users` not false, replace urls by `urls` for each example
        if users != False:
            train.text = train.text.apply(lambda x: USER_PATTERN.sub(users, x))
            test.text = test.text.apply(lambda x: USER_PATTERN.sub(users, x))

        # If `urls` not false, replace urls by `urls` for each example
        if urls != False:
            train.text = train.text.apply(lambda x: URL_PATTERN.sub(urls, x))
            test.text = test.text.apply(lambda x: URL_PATTERN.sub(urls, x))

        # If `spaces` not false, replace double spaces by `spaces` for each example
        if spaces != False:
            train.text = train.text.apply(lambda x: SPACES_PATTERN.sub(spaces, x).strip())
            test.text = test.text.apply(lambda x: SPACES_PATTERN.sub(spaces, x).strip())

        # If `lower` not false, lower each example
        if lower != False:
            train.text = train.text.apply(lambda x: x.lower())
            test.text = test.text.apply(lambda x: x.lower())

        # If `return_sets` is 'splits' return independantly train and test
        if return_sets == 'splits':
            return train, test
        
        full = pd.concat([train, test], ignore_index=True)
        full['split'] = full.example_id.apply(lambda x: x.split('_')[0])

        # If `return_sets` is 'full' return only the train test concatenation
        if return_sets == 'full':
            return full

        # If `return_sets` is 'all' return the train test concatenation and train and test independantly
        return full, train, test

class GoEmotions(DataManager):

    
    train_path = "data/go_emotions/train.jsonl"
    dev_path = "data/go_emotions/dev.jsonl"
    test_path = "data/go_emotions/test.jsonl"

    num_class = 28

    class_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", 
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
    label_2_id = {c:i for i, c in enumerate(class_labels)}
    id_2_label = {i:c for i, c in enumerate(class_labels)}

    # Old way of loading data
    @staticmethod
    def preprocess_data(return_sets : str = 'all'):
        assert return_sets in ['all', 'full', 'splits'], \
            f"return_sets should be set to 'all', 'full' or 'splits'. Given value {return_sets}"
        
        train = read_jsonl(GoEmotions.train_path)
        dev = read_jsonl(GoEmotions.dev_path)
        test = read_jsonl(GoEmotions.test_path)

        # If `return_sets` is 'splits' return independantly train and test
        if return_sets == 'splits':
            return train, dev, test
        
        full = []
        full += [ex | {'split':'train'} for ex in train]
        full += [ex | {'split':'dev'} for ex in dev]
        full += [ex | {'split':'test'} for ex in test]

        # If `return_sets` is 'full' return only the train test concatenation
        if return_sets == 'full':
            return full

        # If `return_sets` is 'all' return the train test concatenation and train and test independantly
        return full, train, dev, test

    @staticmethod
    def parse_label_id(label_id):
        y = [0] * 28
        for l in re.split(',|_', label_id):
            y[int(l)] = 1
        return y
    

MANAGER_CLASS_MAP = {
    "semeval": SemEval,
    "goemotions": GoEmotions,
}
