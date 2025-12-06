import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

CYCLEGAN_PATH = PROJECT_ROOT / "pytorch-CycleGAN-and-pix2pix"

CHECKPOINTS_DIR = CYCLEGAN_PATH / "checkpoints"
RESULTS_DIR = CYCLEGAN_PATH / "results"

DATASETS_DIR = PROJECT_ROOT / "training_datasets"
CONFIG_DIR = PROJECT_ROOT / "config"
ASSETS_DIR = PROJECT_ROOT / "assets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"

PRESETS_FILE = CONFIG_DIR / "training_presets.json"


def setup_cyclegan_path():
    cyclegan_str = str(CYCLEGAN_PATH)
    if cyclegan_str not in sys.path:
        sys.path.insert(0, cyclegan_str)


def ensure_directories():
    dirs_to_create = [
        DATASETS_DIR,
        CONFIG_DIR,
        ASSETS_DIR,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
