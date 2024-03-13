from pathlib import Path
import shutil
import os

root_path = Path(os.environ["HOME"])
results_path = root_path / "irony-categories/results/"

for path in results_path.glob("*/predictions.jsonl"):
    dst = root_path / path.parts[-2]
    if not dst.is_dir():
        dst.mkdir()
    shutil.copy(path, dst / "predictions.jsonl")
    shutil.copy(path.with_name("trainer_state.json"), dst / "trainer_state.json")


