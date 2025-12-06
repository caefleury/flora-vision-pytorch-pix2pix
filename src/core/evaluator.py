import json
import numpy as np
from PIL import Image
from pathlib import Path

from ..utils.paths import CHECKPOINTS_DIR, DATASETS_DIR, RESULTS_DIR
from .detector import LeafDiseaseDetector
from .metrics import calculate_threshold, calculate_accuracy, calculate_statistics


def find_test_dataset_for_model(model_name):
    if not DATASETS_DIR.exists():
        return None

    for dataset_dir in sorted(DATASETS_DIR.iterdir(), reverse=True):
        if dataset_dir.is_dir() and dataset_dir.name.startswith(model_name):
            test_dir = dataset_dir / "test"
            if test_dir.exists() and any(test_dir.iterdir()):
                return test_dir

    return None


def evaluate_model(model_name, epoch="latest", test_dir=None, save_results=True):
    if test_dir is None:
        test_dir = find_test_dataset_for_model(model_name)

    if test_dir is None or not test_dir.exists():
        return {"error": f"Nenhum dataset de teste encontrado para o modelo '{model_name}'"}

    model_dir = CHECKPOINTS_DIR / model_name
    if not model_dir.exists():
        return {"error": f"Modelo '{model_name}' nao encontrado em checkpoints"}

    epoch_file = model_dir / f"{epoch}_net_G.pth"
    if not epoch_file.exists() and epoch != "latest":
        return {"error": f"Epoca '{epoch}' nao encontrada para o modelo '{model_name}'"}

    try:
        detector = LeafDiseaseDetector(model_name, epoch)
    except Exception as e:
        return {"error": f"Erro ao carregar modelo: {str(e)}"}

    results = {"healthy": [], "diseased": [], "metrics": {}}
    scores_list = []

    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

    if not test_images:
        return {"error": f"Nenhuma imagem de teste encontrada em {test_dir}"}

    for img_path in test_images:
        try:
            img = Image.open(img_path)
            result = detector.process_image(img)

            is_healthy = img_path.name.startswith("healthy")
            label = "healthy" if is_healthy else "diseased"

            results[label].append({"filename": img_path.name, "score": result["score"]})

            scores_list.append({
                "image": img_path.stem,
                "ciede2000": result["score"],
                "label": label,
            })
        except Exception as e:
            print(f"Erro ao avaliar {img_path}: {e}")
            continue

    healthy_scores = [r["score"] for r in results["healthy"]]
    diseased_scores = [r["score"] for r in results["diseased"]]

    if healthy_scores:
        stats = calculate_statistics(healthy_scores)
        results["metrics"]["healthy_mean"] = stats["mean"]
        results["metrics"]["healthy_std"] = stats["std"]

    if diseased_scores:
        stats = calculate_statistics(diseased_scores)
        results["metrics"]["diseased_mean"] = stats["mean"]
        results["metrics"]["diseased_std"] = stats["std"]

    if healthy_scores and diseased_scores:
        threshold = calculate_threshold(healthy_scores)
        results["metrics"]["suggested_threshold"] = threshold
        results["metrics"]["accuracy"] = calculate_accuracy(
            healthy_scores, diseased_scores, threshold
        )
        results["metrics"]["total_test_images"] = len(healthy_scores) + len(diseased_scores)

    if save_results and scores_list:
        scores_list.sort(key=lambda x: x["ciede2000"], reverse=True)
        results_dir = RESULTS_DIR / model_name / f"test_{epoch}"
        results_dir.mkdir(parents=True, exist_ok=True)
        scores_file = results_dir / "anomaly_scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores_list, f, indent=2)
        results["saved_to"] = str(scores_file)

    return results


def get_models_without_evaluation():
    models = []

    if not CHECKPOINTS_DIR.exists():
        return models

    for model_dir in CHECKPOINTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        model_files = list(model_dir.glob("*_net_G.pth"))
        if not model_files:
            continue

        epochs = []
        for f in model_files:
            epoch = f.name.replace("_net_G.pth", "")
            epochs.append(epoch)

        has_results = False
        if RESULTS_DIR.exists():
            model_results = RESULTS_DIR / model_name
            if model_results.exists():
                for test_dir in model_results.iterdir():
                    if (test_dir / "anomaly_scores.json").exists():
                        has_results = True
                        break

        test_dir = find_test_dataset_for_model(model_name)

        if not has_results:
            models.append({
                "model_name": model_name,
                "epochs": epochs,
                "has_test_data": test_dir is not None,
                "test_dir": str(test_dir) if test_dir else None,
            })

    return models


def load_evaluation_results(model_name, epoch="latest"):
    scores_file = RESULTS_DIR / model_name / f"test_{epoch}" / "anomaly_scores.json"

    if not scores_file.exists():
        return None

    with open(scores_file, "r") as f:
        return json.load(f)
