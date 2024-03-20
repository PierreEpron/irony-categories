import pandas as pd
import re

from torch.utils.data import DataLoader
import torch

HASHTAG_LABELS_PATTERN = re.compile(r'#(irony|sarcasm)', flags=re.I)
HASHTAG_NOT_PATTERN = re.compile(r'#not', flags=re.I)
URL_PATTERN = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)", flags=re.I)
USER_PATTERN = re.compile(r'@[^\s]+', flags=re.I)
SPACES_PATTERN = re.compile(r'\s+')

def load_semeval_taskb(
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
    train = pd.read_csv('data/sem_eval/train_emoji.tsv', sep='\t', encoding='utf-8')
    test = pd.read_csv('data/sem_eval/test_emoji.tsv', sep='\t', encoding='utf-8')

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


def preprocess_example(tokenizer, example, max_len):

    turns = [
        {"role": "user", "content": example['text'].strip()},
    ]

    input_ids = tokenizer.apply_chat_template(turns)

    if len(input_ids) > max_len:
        return None

    return {
        'example_id': example['example_id'],
        'text': tokenizer.decode(input_ids),
        'label_id': example['label_id'],
    }


def format_labeled_turns(tokenizer, label_id, turns, example):
    turns = [{'role':turn['role'], 'content':turn['content'].format(**example)} for turn in turns[str(label_id)]]
    input_ids = tokenizer.apply_chat_template(turns, return_tensors='pt').to('cuda')
    if input_ids[0][-1] == tokenizer.eos_token_id:
        input_ids = input_ids[...,:-1]
    return input_ids


def preprocess_examples(tokenizer, examples, max_len):

    examples = [preprocess_example(tokenizer, example, max_len) for example in examples.to_dict(orient='records')]
    return [example for example in examples if example]

    # test = [preprocess(example, script_args.max_len) for example in test.to_dict(orient='records')]
    # test = [example for example in test if example]


def collate_key(batch, key):
  return [ex[key] for ex in batch]


def pad_key(batch, key, pad_value):
    collated = collate_key(batch, key)
    max_len = max([len(ex) for ex in collated])
    return [[pad_value] * (max_len - len(ex)) + ex for ex in collated]


def collate(tokenizer, device="cuda", extra_columns=False):
    def wrapped_collate(batch):
        
        collated_batch = {
            'label_id': torch.tensor(collate_key(batch, 'label_id')).to(device),
            'input_ids': torch.LongTensor(pad_key(batch, 'input_ids', tokenizer.pad_token_id)).to(device),
            'attention_mask': torch.LongTensor(pad_key(batch, 'attention_mask', 0)).to(device),
        }

        if extra_columns:

            collated_batch.update({
                'example_id': collate_key(batch, 'example_id'), 'text': collate_key(batch, 'text')
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
        shuffle=shuffle,
        generator=torch.Generator(device='cuda'),
    )



