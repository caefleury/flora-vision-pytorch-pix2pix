import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from skimage import color

from src.core.metrics import calculate_ciede2000, calculate_mean_score


def process_results(results_dir, output_file="anomaly_scores.json"):
    images_dir = (
        os.path.join(results_dir, "images")
        if not results_dir.endswith("images")
        else results_dir
    )

    if not os.path.exists(images_dir):
        print(f"Erro: {images_dir} nao existe")
        return

    files = os.listdir(images_dir)

    # agrupa por nome da imagem
    image_names = set()
    for f in files:
        if f.endswith("_fake_B_rgb.png"):
            image_names.add(f.replace("_fake_B_rgb.png", ""))

    scores = []

    print(f"Processando {len(image_names)} imagens...")
    print("-" * 80)

    for img_name in sorted(image_names):
        fake_path = os.path.join(images_dir, f"{img_name}_fake_B_rgb.png")
        real_path = os.path.join(images_dir, f"{img_name}_real_B_rgb.png")

        if not os.path.exists(fake_path) or not os.path.exists(real_path):
            continue

        fake_img = np.array(Image.open(fake_path)) / 255.0
        real_img = np.array(Image.open(real_path)) / 255.0

        fake_lab = color.rgb2lab(fake_img)
        real_lab = color.rgb2lab(real_img)

        delta_e = calculate_ciede2000(real_lab, fake_lab)
        score = calculate_mean_score(delta_e)

        is_healthy = img_name.lower().startswith("leaf") or img_name.lower().startswith("healthy")
        label = "healthy" if is_healthy else "diseased"

        scores.append({"image": img_name, "ciede2000": float(score), "label": label})
        print(f"{img_name:50s} | CIEDE2000: {score:8.2f} | {label}")

    scores.sort(key=lambda x: x["ciede2000"], reverse=True)

    output_path = os.path.join(
        results_dir if not results_dir.endswith("images") else os.path.dirname(results_dir),
        output_file,
    )
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("-" * 80)
    print(f"\nResultados salvos em: {output_path}")

    healthy_scores = [s["ciede2000"] for s in scores if s["label"] == "healthy"]
    diseased_scores = [s["ciede2000"] for s in scores if s["label"] == "diseased"]

    if healthy_scores:
        print(f"\nImagens saudaveis:")
        print(f"  Quantidade: {len(healthy_scores)}")
        print(f"  Media CIEDE2000: {np.mean(healthy_scores):.2f}")
        print(f"  Desvio: {np.std(healthy_scores):.2f}")

    if diseased_scores:
        print(f"\nImagens doentes:")
        print(f"  Quantidade: {len(diseased_scores)}")
        print(f"  Media CIEDE2000: {np.mean(diseased_scores):.2f}")
        print(f"  Desvio: {np.std(diseased_scores):.2f}")

    if healthy_scores and diseased_scores:
        threshold = np.mean(healthy_scores) + 2 * np.std(healthy_scores)
        print(f"\nLimiar sugerido: {threshold:.2f}")

        correct = sum(
            1 for s in scores
            if (s["ciede2000"] > threshold) == (s["label"] == "diseased")
        )
        accuracy = correct / len(scores) * 100
        print(f"Acuracia: {accuracy:.1f}%")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecta anomalias usando CIEDE2000")
    parser.add_argument("--results_dir", type=str, required=True, help="Diretorio de resultados")
    parser.add_argument("--output", type=str, default="anomaly_scores.json", help="Arquivo JSON de saida")

    args = parser.parse_args()
    process_results(args.results_dir, args.output)
