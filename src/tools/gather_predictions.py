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
        print(HOME_PATH / f"{path.parts[-2]}_{path.parts[-1]}")

    for path in tqdm(list(target_path.glob("**/metrics.csv"))):
        print(HOME_PATH / f"{path.parts[-3]}_{path.parts[-1]}")

# for path in results_path.glob("*/predictions.jsonl"):
#     dst = root_path / path.parts[-2]
#     if not dst.is_dir():
#         dst.mkdir()
#     shutil.copy(path, dst / "predictions.jsonl")
#     shutil.copy(path.with_name("trainer_state.json"), dst / "trainer_state.json")

