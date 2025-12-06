import numpy as np
import matplotlib.pyplot as plt


def create_heatmap_figure(result):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(result["original"])
    axes[0, 0].set_title("Imagem Original", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result["generated"])
    axes[0, 1].set_title("Imagem Reconstruida", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    heatmap = result["heatmap"]
    im = axes[1, 0].imshow(heatmap, cmap="jet", vmin=0, vmax=10)
    axes[1, 0].set_title(
        f'Mapa de Diferencas CIEDE2000\nScore Medio: {result["score"]:.2f}',
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(result["generated"])
    im2 = axes[1, 1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=10)
    axes[1, 1].set_title("Sobreposicao (Anomalias Destacadas)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def create_simple_overlay(result):
    generated = result["generated"].astype(np.float32) / 255.0
    heatmap = result["heatmap"]

    heatmap_norm = np.clip(heatmap / 10.0, 0, 1)

    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap_norm)[:, :, :3]

    alpha = 0.5
    overlay = (1 - alpha) * generated + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def create_comparison_figure(real_img, fake_img, delta_e, title=""):
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

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig
