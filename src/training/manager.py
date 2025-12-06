import os
import sys
import shutil
import subprocess
import threading
import queue
import time
from pathlib import Path
from datetime import datetime

from PIL import Image

from ..utils.paths import CYCLEGAN_PATH, CHECKPOINTS_DIR, DATASETS_DIR
from ..core.evaluator import evaluate_model
from .config import TrainingConfig, TrainingProgress


# classe necessaria para gerenciar estado do treinamento
class TrainingManager:

    def __init__(self):
        self.progress = TrainingProgress()
        self.output_queue = queue.Queue()
        self.training_thread = None
        self.stop_requested = False

    def prepare_dataset(self, healthy_images, diseased_images, dataset_name):

        self.progress.status = "preparing"
        self.progress.message = "Preparando dataset..."

        dataset_path = DATASETS_DIR / dataset_name
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"

        if dataset_path.exists():
            shutil.rmtree(dataset_path)

        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)

        # 80% treino, 20% teste para imagens saudaveis
        num_healthy = len(healthy_images)
        split_idx = int(num_healthy * 0.8)

        for i, (filename, img_bytes) in enumerate(healthy_images):
            try:
                img = Image.open(img_bytes)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                safe_name = f"healthy_{i:04d}.jpg"

                if i < split_idx:
                    img.save(train_path / safe_name, "JPEG", quality=95)
                else:
                    img.save(test_path / safe_name, "JPEG", quality=95)
            except Exception as e:
                print(f"Erro ao processar imagem saudavel {filename}: {e}")
                continue

        # imagens doentes vao apenas para teste
        for i, (filename, img_bytes) in enumerate(diseased_images):
            try:
                img = Image.open(img_bytes)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                safe_name = f"diseased_{i:04d}.jpg"
                img.save(test_path / safe_name, "JPEG", quality=95)
            except Exception as e:
                print(f"Erro ao processar imagem doente {filename}: {e}")
                continue

        self.progress.message = (
            f"Dataset preparado: {split_idx} treino, "
            f"{num_healthy - split_idx + len(diseased_images)} teste"
        )

        return dataset_path

    def build_training_command(self, config, dataroot):
        cmd = [
            sys.executable,
            str(CYCLEGAN_PATH / "train.py"),
            "--dataroot",
            str(dataroot),
            "--name",
            config.name,
            "--model",
            "colorization",
            "--dataset_mode",
            "colorization",
            "--checkpoints_dir",
            str(CHECKPOINTS_DIR),
            # Training params
            "--n_epochs",
            str(config.n_epochs),
            "--n_epochs_decay",
            str(config.n_epochs_decay),
            "--batch_size",
            str(config.batch_size),
            "--lr",
            str(config.lr),
            "--beta1",
            str(config.beta1),
            "--lr_policy",
            config.lr_policy,
            # Network architecture
            "--netG",
            config.netG,
            "--netD",
            config.netD,
            "--ngf",
            str(config.ngf),
            "--ndf",
            str(config.ndf),
            "--n_layers_D",
            str(config.n_layers_D),
            "--norm",
            config.norm,
            "--load_size",
            str(config.load_size),
            "--crop_size",
            str(config.crop_size),
            "--preprocess",
            config.preprocess,
            "--gan_mode",
            config.gan_mode,
            "--save_epoch_freq",
            str(config.save_epoch_freq),
            "--display_freq",
            str(config.display_freq),
            "--print_freq",
            str(config.print_freq),
            "--no_html",
        ]

        if config.no_flip:
            cmd.append("--no_flip")

        return cmd

    def _parse_training_output(self, line):
        line = line.strip()

        if "End of epoch" in line:
            try:
                parts = line.split()
                epoch_idx = parts.index("epoch") + 1
                current = int(parts[epoch_idx])
                total = int(parts[epoch_idx + 2])
                self.progress.current_epoch = current
                self.progress.total_epochs = total
            except (ValueError, IndexError):
                pass

        elif "G_GAN:" in line or "G_L1:" in line:
            try:
                if "(epoch:" in line:
                    epoch_part = line.split("(epoch:")[1].split(",")[0].strip()
                    self.progress.current_epoch = int(epoch_part)
                if "iters:" in line:
                    iter_part = line.split("iters:")[1].split(",")[0].strip()
                    self.progress.current_iter = int(iter_part)

                losses = {}
                for part in line.split():
                    if ":" in part and not part.startswith("("):
                        try:
                            key, val = part.split(":")
                            if key in ["G_GAN", "G_L1", "D_real", "D_fake"]:
                                losses[key] = float(val)
                        except ValueError:
                            continue
                if losses:
                    self.progress.losses = losses
            except (ValueError, IndexError):
                pass

        elif "training images" in line:
            try:
                num = int(line.split("=")[1].strip())
                self.progress.message = f"Iniciando treino com {num} imagens"
            except (ValueError, IndexError):
                pass

        if "saving" in line.lower():
            self.progress.message = line

    def _training_worker(self, cmd, dataroot, config):
        try:
            self.progress.status = "training"
            self.progress.total_epochs = config.n_epochs + config.n_epochs_decay
            self.progress.start_time = time.time()
            self.progress.message = "Iniciando treinamento..."

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(CYCLEGAN_PATH),
            )

            for line in iter(process.stdout.readline, ""):
                if self.stop_requested:
                    process.terminate()
                    self.progress.status = "error"
                    self.progress.error = "Treinamento cancelado pelo usuário"
                    return

                self._parse_training_output(line)
                self.progress.elapsed_time = time.time() - self.progress.start_time
                self.output_queue.put(line)

            process.wait()

            if process.returncode != 0:
                self.progress.status = "error"
                self.progress.error = f"Treinamento falhou com codigo {process.returncode}"
                return

            # avalia o modelo apos treino
            self.progress.status = "evaluating"
            self.progress.message = "Avaliando modelo..."

            evaluation = evaluate_model(config.name, "latest", dataroot / "test")
            self.progress.evaluation_results = evaluation

            self.progress.status = "completed"
            self.progress.message = "Treinamento concluido!"

        except Exception as e:
            self.progress.status = "error"
            self.progress.error = str(e)

    def start_training(self, healthy_images, diseased_images, config):
        if self.training_thread and self.training_thread.is_alive():
            return False

        self.progress = TrainingProgress()
        self.stop_requested = False

        while not self.output_queue.empty():
            self.output_queue.get()

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{config.name}_{timestamp}"
            dataroot = self.prepare_dataset(
                healthy_images, diseased_images, dataset_name
            )
        except Exception as e:
            self.progress.status = "error"
            self.progress.error = f"Erro ao preparar dataset: {e}"
            return False

        cmd = self.build_training_command(config, dataroot)

        self.training_thread = threading.Thread(
            target=self._training_worker, args=(cmd, dataroot, config)
        )
        self.training_thread.start()

        return True

    def stop_training(self):
        self.stop_requested = True

    def get_progress(self):
        return self.progress

    def reset(self):
        self.progress = TrainingProgress()
        self.stop_requested = False
        self.training_thread = None
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def get_output_lines(self, max_lines=100):
        lines = []
        while not self.output_queue.empty() and len(lines) < max_lines:
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def is_training(self):
        return self.training_thread is not None and self.training_thread.is_alive()


_training_manager = None


def get_training_manager():
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager
