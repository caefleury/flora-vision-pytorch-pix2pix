import json

from ..utils.paths import PRESETS_FILE
from .config import TrainingConfig


def load_training_presets():
    if PRESETS_FILE.exists():
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_training_presets():
    return {
        "rapido": TrainingConfig(
            name="leaf_fast",
            n_epochs=25,
            n_epochs_decay=25,
            batch_size=4,
            save_epoch_freq=10,
        ),
        "padrao": TrainingConfig(
            name="leaf_standard",
            n_epochs=50,
            n_epochs_decay=50,
            batch_size=1,
            save_epoch_freq=10,
        ),
        "completo": TrainingConfig(
            name="leaf_complete",
            n_epochs=100,
            n_epochs_decay=100,
            batch_size=1,
            save_epoch_freq=20,
        ),
    }


def get_dataset_category(num_images):
    presets_data = load_training_presets()
    categories = presets_data.get("dataset_categories", {})

    for category_name, category_info in categories.items():
        min_img = category_info.get("min_images", 0)
        max_img = category_info.get("max_images", float("inf"))
        if min_img <= num_images <= max_img:
            return category_name
    return None


def get_recommended_presets(num_images):
    presets_data = load_training_presets()
    categories = presets_data.get("dataset_categories", {})

    category_name = get_dataset_category(num_images)

    if category_name and category_name in categories:
        category = categories[category_name]
        return {
            "category": category_name,
            "description": category.get("description", ""),
            "presets": category.get("presets", {}),
            "tips": presets_data.get("tips", {}),
        }

    return {
        "category": "pequeno",
        "description": "Configuracao padrao",
        "presets": {},
        "tips": presets_data.get("tips", {}),
    }


def create_config_from_preset(preset_flags, model_name="leaf_model"):
    return TrainingConfig(
        name=model_name,
        n_epochs=preset_flags.get("n_epochs", 50),
        n_epochs_decay=preset_flags.get("n_epochs_decay", 50),
        batch_size=preset_flags.get("batch_size", 1),
        lr=preset_flags.get("lr", 0.0002),
        beta1=preset_flags.get("beta1", 0.5),
        lr_policy=preset_flags.get("lr_policy", "linear"),
        netG=preset_flags.get("netG", "unet_256"),
        netD=preset_flags.get("netD", "basic"),
        ngf=preset_flags.get("ngf", 64),
        ndf=preset_flags.get("ndf", 64),
        n_layers_D=preset_flags.get("n_layers_D", 3),
        norm=preset_flags.get("norm", "batch"),
        load_size=preset_flags.get("load_size", 286),
        crop_size=preset_flags.get("crop_size", 256),
        preprocess=preset_flags.get("preprocess", "resize_and_crop"),
        no_flip=preset_flags.get("no_flip", False),
        gan_mode=preset_flags.get("gan_mode", "lsgan"),
        save_epoch_freq=preset_flags.get("save_epoch_freq", 10),
    )
