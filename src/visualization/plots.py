import numpy as np
import matplotlib.pyplot as plt


def create_score_distribution_plot(healthy_scores, diseased_scores, threshold=None, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(healthy_scores, bins=20, alpha=0.6, label="Saudaveis", color="#2E7D32")
    ax.hist(diseased_scores, bins=20, alpha=0.6, label="Doentes", color="#C62828")

    if threshold is not None:
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Limiar ({threshold:.2f})",
        )

    ax.set_xlabel("Score CIEDE2000")
    ax.set_ylabel("Frequencia")
    ax.legend()

    plt.tight_layout()
    return fig


def create_comparison_table(results):
    if not results:
        return "Nenhum resultado para comparar."

    header = f"{'Modelo':<30} | {'Acuracia':>10} | {'Separacao':>10} | {'Limiar':>10}"
    separator = "-" * len(header)

    lines = [header, separator]

    for r in results:
        name = r.get("name", "Unknown")[:30]
        acc = r.get("accuracy", 0)
        sep = r.get("separation", 0)
        thresh = r.get("threshold", 0)
        lines.append(f"{name:<30} | {acc:>9.1f}% | {sep:>10.3f} | {thresh:>10.3f}")

    return "\n".join(lines)


def create_metrics_summary_plot(metrics, figsize=(10, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    categories = ["Saudaveis", "Doentes"]
    means = [metrics.get("healthy_mean", 0), metrics.get("diseased_mean", 0)]
    stds = [metrics.get("healthy_std", 0), metrics.get("diseased_std", 0)]
    colors = ["#2E7D32", "#C62828"]

    bars = axes[0].bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    axes[0].set_ylabel("Score CIEDE2000 Medio")
    axes[0].set_title("Comparacao de Scores")

    if "suggested_threshold" in metrics:
        axes[0].axhline(
            metrics["suggested_threshold"],
            color="black",
            linestyle="--",
            label=f"Limiar: {metrics['suggested_threshold']:.2f}",
        )
        axes[0].legend()

    accuracy = metrics.get("accuracy", 0)
    sizes = [accuracy, 100 - accuracy]
    labels = [f"Correto\n{accuracy:.1f}%", f"Incorreto\n{100-accuracy:.1f}%"]
    colors_pie = ["#4CAF50", "#f44336"]

    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct="", startangle=90)
    axes[1].set_title("Acuracia do Modelo")

    plt.tight_layout()
    return fig


def create_training_loss_plot(losses_history, figsize=(10, 4)):
    if not losses_history:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sem dados de treinamento", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    loss_names = list(losses_history[0].keys())

    for loss_name in loss_names:
        values = [h.get(loss_name, 0) for h in losses_history]
        ax.plot(values, label=loss_name, alpha=0.7)

    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Loss")
    ax.set_title("Curva de Treinamento")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
