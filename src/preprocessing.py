import pandas as pd
import re

HASHTAG_PATTERN = re.compile(r'#(irony|sarcasm|not)', flags=re.I)
URL_PATTERN = re.compile("^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$", flags=re.I)
USER_PATTERN = re.compile(r'@[^\s]+', flags=re.I)
SPACES_PATTERN = re.compile(r'\s+')



def load_semeval_taskb(
        return_sets : str = 'all', 
        hashtags = '', 
        users='@user', 
        urls = '', 
        spaces = ' ', 
        lower = ''
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

    # If `hashtags` not false, replace #irony #sarcasm #not by `hashtags` for each example
    if hashtags:
        train.text = train.text.apply(lambda x: HASHTAG_PATTERN.sub(hashtags, x))
        test.text = test.text.apply(lambda x: HASHTAG_PATTERN.sub(hashtags, x))

    # If `users` not false, replace urls by `urls` for each example
    if users:
        train.text = train.text.apply(lambda x: USER_PATTERN.sub(users, x))
        test.text = test.text.apply(lambda x: USER_PATTERN.sub(users, x))

    # If `urls` not false, replace urls by `urls` for each example
    if urls:
        train.text = train.text.apply(lambda x: URL_PATTERN.sub(urls, x))
        test.text = test.text.apply(lambda x: URL_PATTERN.sub(urls, x))

    # If `spaces` not false, replace double spaces by `spaces` for each example
    if spaces:
        train.text = train.text.apply(lambda x: SPACES_PATTERN.sub(spaces, x))
        test.text = test.text.apply(lambda x: SPACES_PATTERN.sub(spaces, x))

    # If `lower` not false, lower each example
    if lower:
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


