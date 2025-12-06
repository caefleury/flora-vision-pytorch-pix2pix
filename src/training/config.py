from dataclasses import dataclass, field


# dataclasses exigem anotacao de tipo
@dataclass
class TrainingConfig:
    name: str = "leaf_model"
    n_epochs: int = 50
    n_epochs_decay: int = 50
    batch_size: int = 1
    lr: float = 0.0002
    beta1: float = 0.5
    lr_policy: str = "linear"
    netG: str = "unet_256"
    netD: str = "basic"
    ngf: int = 64
    ndf: int = 64
    n_layers_D: int = 3
    norm: str = "batch"
    load_size: int = 286
    crop_size: int = 256
    no_flip: bool = False
    preprocess: str = "resize_and_crop"
    gan_mode: str = "lsgan"
    save_epoch_freq: int = 10
    display_freq: int = 400
    print_freq: int = 100


@dataclass
class TrainingProgress:
    status: str = "idle"
    current_epoch: int = 0
    total_epochs: int = 0
    current_iter: int = 0
    total_iters: int = 0
    losses: dict = field(default_factory=dict)
    message: str = ""
    error: str = None
    evaluation_results: dict = None
    start_time: float = None
    elapsed_time: float = 0.0
