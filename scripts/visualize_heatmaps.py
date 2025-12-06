import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color

from src.core.metrics import calculate_ciede2000


def create_heatmap_visualization(fake_path, real_path, output_path):
    fake_img = np.array(Image.open(fake_path)) / 255.0
    real_img = np.array(Image.open(real_path)) / 255.0

    fake_lab = color.rgb2lab(fake_img)
    real_lab = color.rgb2lab(real_img)

    delta_e = calculate_ciede2000(real_lab, fake_lab)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(real_img)
    axes[0, 0].set_title("Original", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(fake_img)
    axes[0, 1].set_title("Colorizado", fontsize=12)
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(delta_e, cmap="jet", vmin=0, vmax=10)
    axes[1, 0].set_title(f"Heatmap CIEDE2000\nMedia: {np.mean(delta_e):.2f}", fontsize=12)
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(fake_img)
    im2 = axes[1, 1].imshow(delta_e, cmap="jet", alpha=0.5, vmin=0, vmax=10)
    axes[1, 1].set_title("Sobreposicao", fontsize=12)
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return np.mean(delta_e)


def process_images(results_dir, num_images=10, output_dir="heatmaps"):
    scores_path = os.path.join(results_dir, "anomaly_scores.json")

    with open(scores_path, "r") as f:
        scores = json.load(f)

    images_dir = os.path.join(results_dir, "images")
    output_path = os.path.join(results_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    print(f"Criando visualizacoes de heatmap...")
    print(f"Diretorio de saida: {output_path}")
    print("-" * 80)

    diseased = [s for s in scores if s["label"] == "diseased"]
    healthy = [s for s in scores if s["label"] == "healthy"]

    print(f"\nProcessando top {num_images} imagens doentes...")
    for i, item in enumerate(diseased[:num_images]):
        img_name = item["image"]
        score = item["ciede2000"]

        fake_path = os.path.join(images_dir, f"{img_name}_fake_B_rgb.png")
        real_path = os.path.join(images_dir, f"{img_name}_real_B_rgb.png")
        output_file = os.path.join(
            output_path,
            f"doente_{i+1:02d}_{score:.2f}_{img_name.replace(' ', '_')}.png",
        )

        if os.path.exists(fake_path) and os.path.exists(real_path):
            create_heatmap_visualization(fake_path, real_path, output_file)
            print(f"  {i+1}. {img_name:50s} | CIEDE2000: {score:.2f}")

    print(f"\nProcessando top {num_images} imagens saudaveis...")
    healthy_sorted = sorted(healthy, key=lambda x: x["ciede2000"])
    for i, item in enumerate(healthy_sorted[:num_images]):
        img_name = item["image"]
        score = item["ciede2000"]

        fake_path = os.path.join(images_dir, f"{img_name}_fake_B_rgb.png")
        real_path = os.path.join(images_dir, f"{img_name}_real_B_rgb.png")
        output_file = os.path.join(
            output_path,
            f"saudavel_{i+1:02d}_{score:.2f}_{img_name.replace(' ', '_')}.png",
        )

        if os.path.exists(fake_path) and os.path.exists(real_path):
            create_heatmap_visualization(fake_path, real_path, output_file)
            print(f"  {i+1}. {img_name:50s} | CIEDE2000: {score:.2f}")

    print("-" * 80)
    print(f"\nHeatmaps salvos em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza heatmaps CIEDE2000")
    parser.add_argument("--results_dir", type=str, required=True, help="Diretorio de resultados")
    parser.add_argument("--num_images", type=int, default=10, help="Numero de imagens por categoria")

    args = parser.parse_args()
    process_images(args.results_dir, num_images=args.num_images)
