import numpy as np
from skimage import color
from skimage.color import deltaE_ciede2000


def calculate_ciede2000(real_lab, fake_lab):
    return deltaE_ciede2000(real_lab, fake_lab)


def calculate_mean_score(delta_e):
    return float(np.mean(delta_e))


def calculate_threshold(healthy_scores, n_std=2.0):
    if not healthy_scores:
        return 3.0
    return float(np.mean(healthy_scores) + n_std * np.std(healthy_scores))


def calculate_accuracy(healthy_scores, diseased_scores, threshold):
    correct = 0
    total = len(healthy_scores) + len(diseased_scores)

    if total == 0:
        return 0.0

    for score in healthy_scores:
        if score <= threshold:
            correct += 1

    for score in diseased_scores:
        if score > threshold:
            correct += 1

    return float(correct / total * 100)


def calculate_statistics(scores):
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def rgb_to_lab(rgb_image):
    # normaliza para 0-1 se necessario
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    return color.rgb2lab(rgb_image)


def lab_to_rgb(lab_image):
    rgb = color.lab2rgb(lab_image)
    return (rgb * 255).astype(np.uint8)
