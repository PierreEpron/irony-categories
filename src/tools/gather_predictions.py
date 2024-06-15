from dataclasses import dataclass, field
from pathlib import Path
import shutil
import os

from transformers import HfArgumentParser
from tqdm import tqdm


HOME_PATH = Path(os.environ["HOME"])


@dataclass
class ScriptArguments:
    target_path: str = field(metadata={"help":"Gather predictions and metrics inside the target folder."})


if __name__ == "__main__":

    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]

    target_path = Path(script_args.target_path)

    for path in tqdm(list(target_path.glob("**/predictions.jsonl"))):
        shutil.copy(path, HOME_PATH / f"{path.parts[-2]}_{path.parts[-1]}")

    for path in tqdm(list(target_path.glob("**/metrics.csv"))):
        shutil.copy(path, HOME_PATH / f"{path.parts[-3]}_{path.parts[-1]}")