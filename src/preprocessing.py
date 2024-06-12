import pandas as pd
import re

from torch.utils.data import DataLoader
from datasets import Dataset
import torch

from src.utils import read_jsonl

HASHTAG_LABELS_PATTERN = re.compile(r'#(irony|sarcasm)', flags=re.I)
HASHTAG_NOT_PATTERN = re.compile(r'#not', flags=re.I)
URL_PATTERN = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)", flags=re.I)
USER_PATTERN = re.compile(r'@[^\s]+', flags=re.I)
SPACES_PATTERN = re.compile(r'\s+')

class SemEval:
    
    train_path = "data/sem_eval/train_emoji.tsv"
    test_path = "data/sem_eval/test_emoji.tsv"
    
    class_labels = ["No Irony", "Irony by Clash", "Situational Irony", "Other Irony"]
    label_2_id = {c:i for i, c in enumerate(class_labels)}
    id_2_label = {i:c for i, c in enumerate(class_labels)}

    @staticmethod
    def load_data(
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
            train.text = train.text.apply(lambda x: HASHTAG_LABELS_PATTERN.sub(hashtag_labels, x))
            test.text = test.text.apply(lambda x: HASHTAG_LABELS_PATTERN.sub(hashtag_labels, x))

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

class GoEmotions:
    
    train_path = "data/go_emotions/train.jsonl"
    dev_path = "data/go_emotions/dev.jsonl"
    test_path = "data/go_emotions/test.jsonl"
    
    class_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", 
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
    label_2_id = {c:i for i, c in enumerate(class_labels)}
    id_2_label = {i:c for i, c in enumerate(class_labels)}

    @staticmethod
    def load_data(return_sets : str = 'all'):
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


def tokenize_example(example, tokenizer, turns, num_classes=4):
    
    # Copy and format turns
    turns = [{'role':turn['role'], 'content':turn['content'].format(**example)} for turn in turns]

    # Apply chat template
    input_ids = tokenizer.apply_chat_template(turns)
    example['input_ids'] = input_ids

    # Create attention mask
    example['attention_mask'] = [1] * len(input_ids)

    # One hot labels if they doesn't exist
    if 'labels' not in example:
        example['labels'] = torch.nn.functional.one_hot(
            torch.tensor(example['label_id']), 
            num_classes=num_classes
        )

    return example

def make_dataset(examples, tokenizer, turns, max_len=105, num_classes=4):

    if isinstance(examples, pd.DataFrame):
        examples = examples.to_dict(orient="records")

    examples = Dataset.from_list(examples)
    examples = examples.map(lambda x: tokenize_example(x, tokenizer, turns, num_classes))

    if max_len > 0:
        examples = examples.filter(lambda x: len(x['input_ids']) < max_len)
        
    return examples

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

def make_loader(dataset, tokenizer, batch_size, extra_columns=False, shuffle=True):
    '''
        Create dataset, tokenize examples, filter example by max_len, pad examples then return a loader.
    '''
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate(tokenizer, extra_columns=extra_columns), 
        shuffle=shuffle
    )