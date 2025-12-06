import torch
import numpy as np
from PIL import Image
from skimage import color
import torchvision.transforms as transforms
from pathlib import Path
from argparse import Namespace

from ..utils.paths import CHECKPOINTS_DIR, setup_cyclegan_path
from .metrics import calculate_ciede2000, calculate_mean_score

setup_cyclegan_path()
from models import create_model


def get_available_models(checkpoints_dir=None):
    if checkpoints_dir is None:
        checkpoints_path = CHECKPOINTS_DIR
    else:
        checkpoints_path = Path(checkpoints_dir)

    if not checkpoints_path.exists():
        return []

    models = []
    for item in checkpoints_path.iterdir():
        if item.is_dir():
            has_model = any(
                f.name.endswith("_net_G.pth") for f in item.iterdir() if f.is_file()
            )
            if has_model:
                models.append(item.name)

    return sorted(models)


def get_available_epochs(model_name, checkpoints_dir=None):
    if checkpoints_dir is None:
        checkpoints_path = CHECKPOINTS_DIR
    else:
        checkpoints_path = Path(checkpoints_dir)

    model_dir = checkpoints_path / model_name
    if not model_dir.exists():
        return []

    epochs = []
    for f in model_dir.iterdir():
        if f.is_file() and f.name.endswith("_net_G.pth"):
            epoch = f.name.replace("_net_G.pth", "")
            epochs.append(epoch)

    def sort_key(x):
        if x == "latest":
            return (1, 0)
        try:
            return (0, int(x))
        except ValueError:
            return (0, 0)

    return sorted(epochs, key=sort_key)


def create_inference_options(model_name, epoch="latest", checkpoints_dir=None):
    if checkpoints_dir is None:
        checkpoints_dir = str(CHECKPOINTS_DIR)

    opt = Namespace(
        model="colorization",
        name=model_name,
        checkpoints_dir=checkpoints_dir,
        epoch=epoch,
        load_iter=0,
        input_nc=1,
        output_nc=2,
        ngf=64,
        ndf=64,
        netG="unet_256",
        netD="basic",
        n_layers_D=3,
        norm="batch",
        init_type="normal",
        init_gain=0.02,
        no_dropout=False,
        dataset_mode="colorization",
        direction="AtoB",
        preprocess="resize_and_crop",
        load_size=256,
        crop_size=256,
        isTrain=False,
        verbose=False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        gan_mode="vanilla",
    )

    return opt


# classe necessaria para manter estado do modelo carregado
class LeafDiseaseDetector:

    def __init__(self, model_name, epoch="latest", checkpoints_dir=None):
        self.model_name = model_name
        self.epoch = epoch
        self.opt = create_inference_options(model_name, epoch, checkpoints_dir)

        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.eval()

        self.transform = transforms.Compose([transforms.Resize((256, 256))])

    def preprocess_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)

        im = np.array(image)
        lab = color.rgb2lab(im).astype(np.float32)

        lab_t = transforms.ToTensor()(lab)

        # normaliza L para [-1, 1] e ab para [-1, 1]
        A = lab_t[[0], ...] / 50.0 - 1.0
        B = lab_t[[1, 2], ...] / 110.0

        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {"A": A, "B": B, "A_paths": "", "B_paths": ""}

    def lab2rgb(self, L, AB):
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb.astype(np.uint8)

    def process_image(self, image):
        data = self.preprocess_image(image)

        data["A"] = data["A"].to(self.opt.device)
        data["B"] = data["B"].to(self.opt.device)

        self.model.set_input(data)
        self.model.test()

        real_A = self.model.real_A
        real_B = self.model.real_B
        fake_B = self.model.fake_B

        real_rgb = self.lab2rgb(real_A, real_B)
        fake_rgb = self.lab2rgb(real_A, fake_B)

        real_lab = color.rgb2lab(real_rgb / 255.0)
        fake_lab = color.rgb2lab(fake_rgb / 255.0)
        delta_e = calculate_ciede2000(real_lab, fake_lab)

        mean_score = calculate_mean_score(delta_e)

        threshold = 3.0
        is_diseased = mean_score > threshold

        return {
            "original": real_rgb,
            "generated": fake_rgb,
            "heatmap": delta_e,
            "score": float(mean_score),
            "is_diseased": is_diseased,
            "threshold": threshold,
        }


def process_single_image(image_path, model_name, epoch="latest"):
    detector = LeafDiseaseDetector(model_name, epoch)
    image = Image.open(image_path)
    return detector.process_image(image)
