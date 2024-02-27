from pathlib import Path
import torch
import time
import json
import os


def read_jsonl(path, encoding='utf-8'):
    """
        Shortcut for read jsonl file

        Parameters
        ----------
        path : str or Path, path of file to read.
        encoding : str, default='utf-8', encoding format to use.
    """
    path = Path(path) if isinstance(path, str) else path
    return [json.loads(line) for line in path.read_text(encoding=encoding).strip().split('\n')]


def write_jsonl(path, data, encoding='utf-8'):
    """
        Shortcut for write jsonl file

        Parameters
        ----------
        path : str or Path, path of file to write.
        data : List, list of json data to write.
        encoding : str, default='utf-8', encoding format to use.
    """
    path = Path(path) if isinstance(path, str) else path
    path.write_text('\n'.join([json.dumps(item) for item in data]), encoding=encoding)