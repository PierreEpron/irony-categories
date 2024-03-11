from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import HfArgumentParser

from src.preprocessing import load_semeval_taskb
from src.utils import read_jsonl, write_jsonl

@dataclass
class SplitArguments:
    output_file: str = field(metadata={"help": "the path used to store split index."})
    target_column: Optional[str] = field(default="label_id", metadata={"help": "the column used for stratification."})
    n_splits: Optional[int] = field(default=5, metadata={"help": "the number of split to return."})
    test_size: Optional[int] = field(default=784, metadata={"help": "the size of the test set. If float, used as ratio. If int used as absolute value."})


def get_split(split_id, splits_path, dataset):
    split = read_jsonl(splits_path)[split_id]
    return dataset.loc[split['train']], dataset.loc[split['test']] # test -> val

def split_data(output_file, target_column, n_splits, test_size):
    train, _ = load_semeval_taskb(return_sets='splits')
    y = train[target_column]
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    splits = [{'train':train.tolist(), 'test':test.tolist()} for train, test in sss.split(train, y)]
    write_jsonl(output_file, splits)

if __name__ == "__main__":
    parser = HfArgumentParser([SplitArguments])
    split_args = parser.parse_args_into_dataclasses()[0]        
    split_data(**split_args.__dict__)