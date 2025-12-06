import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.paths import RESULTS_DIR


def analyze_results(json_path, name):
    with open(json_path, "r") as f:
        data = json.load(f)

    healthy = [d["ciede2000"] for d in data if d["label"] == "healthy"]
    diseased = [d["ciede2000"] for d in data if d["label"] == "diseased"]

    if not healthy or not diseased:
        print(f"Aviso: Dados faltando em {json_path}")
        return None

    threshold = np.mean(healthy) + 2 * np.std(healthy)
    correct = sum(
        1 for d in data if (d["ciede2000"] > threshold) == (d["label"] == "diseased")
    )
    accuracy = correct / len(data) * 100

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Imagens saudaveis:")
    print(f"  Quantidade: {len(healthy)}")
    print(f"  Media:  {np.mean(healthy):.3f}")
    print(f"  Desvio: {np.std(healthy):.3f}")
    print(f"  Range:  {np.min(healthy):.3f} - {np.max(healthy):.3f}")

    print(f"\nImagens doentes:")
    print(f"  Quantidade: {len(diseased)}")
    print(f"  Media:  {np.mean(diseased):.3f}")
    print(f"  Desvio: {np.std(diseased):.3f}")
    print(f"  Range:  {np.min(diseased):.3f} - {np.max(diseased):.3f}")

    print(f"\nMetricas:")
    print(f"  Separacao: {np.mean(diseased) - np.mean(healthy):.3f}")
    print(f"  Limiar:    {threshold:.3f}")
    print(f"  Acuracia:  {accuracy:.1f}%")

    return {
        "name": name,
        "healthy_mean": np.mean(healthy),
        "diseased_mean": np.mean(diseased),
        "separation": np.mean(diseased) - np.mean(healthy),
        "accuracy": accuracy,
        "threshold": threshold,
    }


def find_all_results():
    results = []

    if not RESULTS_DIR.exists():
        print(f"Diretorio de resultados nao encontrado: {RESULTS_DIR}")
        return results

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for test_dir in model_dir.iterdir():
            if not test_dir.is_dir():
                continue

            scores_file = test_dir / "anomaly_scores.json"
            if scores_file.exists():
                results.append({
                    "path": str(scores_file),
                    "model": model_dir.name,
                    "epoch": test_dir.name.replace("test_", ""),
                })

    return results


def main():
    available_results = find_all_results()

    if not available_results:
        print("Nenhum resultado de treinamento encontrado.")
        print(f"Procurando em: {RESULTS_DIR}")
        return

    print(f"Encontrados {len(available_results)} arquivos de resultado")

    analyzed_results = []
    for result in available_results:
        name = f"{result['model']} ({result['epoch']})"
        analysis = analyze_results(result["path"], name)
        if analysis:
            analyzed_results.append(analysis)

    if len(analyzed_results) < 2:
        print("\nPrecisa de pelo menos 2 resultados para comparar.")
        return

    # resumo da comparacao
    print(f"\n{'='*80}")
    print(f"  RESUMO DA COMPARACAO")
    print(f"{'='*80}")

    analyzed_results.sort(key=lambda x: x["accuracy"], reverse=True)

    header = f"{'Modelo':<40} | {'Acuracia':>10} | {'Separacao':>10} | {'Limiar':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for result in analyzed_results:
        print(
            f"{result['name']:<40} | {result['accuracy']:>9.1f}% | {result['separation']:>10.3f} | {result['threshold']:>10.3f}"
        )

    best = analyzed_results[0]
    print(f"\n{'='*80}")
    print(f"MELHOR MODELO: {best['name']} (Acuracia: {best['accuracy']:.1f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
