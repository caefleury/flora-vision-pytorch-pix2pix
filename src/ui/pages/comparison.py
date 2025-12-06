import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from ...utils.paths import RESULTS_DIR, CHECKPOINTS_DIR
from ...core.evaluator import evaluate_model, get_models_without_evaluation


def get_model_results():
    models_data = {}

    if RESULTS_DIR.exists():
        for model_dir in RESULTS_DIR.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for test_dir in model_dir.iterdir():
                if not test_dir.is_dir():
                    continue

                scores_file = test_dir / "anomaly_scores.json"
                if not scores_file.exists():
                    continue

                try:
                    with open(scores_file, "r") as f:
                        scores = json.load(f)

                    # Calculate metrics
                    healthy_scores = [
                        s["ciede2000"] for s in scores if s["label"] == "healthy"
                    ]
                    diseased_scores = [
                        s["ciede2000"] for s in scores if s["label"] == "diseased"
                    ]

                    if not healthy_scores or not diseased_scores:
                        continue

                    # Calculate threshold and accuracy
                    healthy_mean = np.mean(healthy_scores)
                    healthy_std = np.std(healthy_scores)
                    diseased_mean = np.mean(diseased_scores)
                    diseased_std = np.std(diseased_scores)

                    # Suggested threshold: mean + 2*std of healthy
                    threshold = healthy_mean + 2 * healthy_std

                    # Calculate accuracy
                    correct = 0
                    total = len(healthy_scores) + len(diseased_scores)

                    for score in healthy_scores:
                        if score <= threshold:
                            correct += 1
                    for score in diseased_scores:
                        if score > threshold:
                            correct += 1

                    accuracy = (correct / total) * 100 if total > 0 else 0

                    # Store results
                    epoch_name = test_dir.name.replace("test_", "")
                    model_key = f"{model_name} ({epoch_name})"

                    # Check if checkpoint exists
                    has_checkpoint = (CHECKPOINTS_DIR / model_name).exists()

                    models_data[model_key] = {
                        "model_name": model_name,
                        "epoch": epoch_name,
                        "accuracy": accuracy,
                        "threshold": threshold,
                        "healthy_mean": healthy_mean,
                        "healthy_std": healthy_std,
                        "diseased_mean": diseased_mean,
                        "diseased_std": diseased_std,
                        "healthy_count": len(healthy_scores),
                        "diseased_count": len(diseased_scores),
                        "total_images": total,
                        "healthy_scores": healthy_scores,
                        "diseased_scores": diseased_scores,
                        "has_checkpoint": has_checkpoint,
                        "has_results": True,
                    }

                except Exception as e:
                    print(f"Error loading {scores_file}: {e}")
                    continue

    # Then, add models from checkpoints that don't have results yet
    if CHECKPOINTS_DIR.exists():
        for model_dir in CHECKPOINTS_DIR.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Check if this model already has results
            already_has_results = any(
                d["model_name"] == model_name for d in models_data.values()
            )

            if not already_has_results:
                # Check if it has model files
                model_files = list(model_dir.glob("*_net_G.pth"))
                if model_files:
                    epochs = []
                    for f in model_files:
                        epoch = f.name.replace("_net_G.pth", "")
                        epochs.append(epoch)

                    epoch_str = ", ".join(sorted(epochs)) if epochs else "N/A"
                    model_key = f"{model_name} (sem avaliação)"

                    models_data[model_key] = {
                        "model_name": model_name,
                        "epoch": epoch_str,
                        "accuracy": None,
                        "threshold": None,
                        "healthy_mean": None,
                        "healthy_std": None,
                        "diseased_mean": None,
                        "diseased_std": None,
                        "healthy_count": 0,
                        "diseased_count": 0,
                        "total_images": 0,
                        "healthy_scores": [],
                        "diseased_scores": [],
                        "has_checkpoint": True,
                        "has_results": False,
                        "available_epochs": epochs,
                    }

    return models_data


def render():

    st.title("Modelos Treinados")
    st.caption("Visualize e compare os modelos treinados")

    models_data = get_model_results()

    if not models_data:
        st.warning("Nenhum modelo encontrado.")
        st.info("Treine um modelo primeiro na aba de treinamento.")
        return

    models_with_results = {
        k: v for k, v in models_data.items() if v.get("has_results", False)
    }
    models_without_results = {
        k: v for k, v in models_data.items() if not v.get("has_results", False)
    }

    # Models with evaluation
    if models_with_results:
        st.header("Modelos Avaliados")

        summary_data = []
        for model_key, data in models_with_results.items():
            summary_data.append(
                {
                    "Modelo": data["model_name"],
                    "Época": data["epoch"],
                    "Acurácia (%)": f"{data['accuracy']:.1f}",
                    "Limiar": f"{data['threshold']:.2f}",
                    "Imagens": data["total_images"],
                }
            )

        summary_data.sort(
            key=lambda x: float(x["Acurácia (%)"].replace(",", ".")), reverse=True
        )

        st.dataframe(summary_data, use_container_width=True, hide_index=True)

        best_model = summary_data[0]
        st.success(
            f"Melhor modelo: {best_model['Modelo']} com {best_model['Acurácia (%)']}% de acurácia"
        )
    else:
        st.info("Nenhum modelo avaliado ainda.")

    # Models without evaluation
    if models_without_results:
        st.divider()
        st.header("Modelos Pendentes")
        st.caption("Modelos com checkpoints mas sem avaliação")

        for model_key, data in models_without_results.items():
            model_name = data["model_name"]
            available_epochs = data.get("available_epochs", [])

            with st.expander(f"{model_name} - Épocas: {data.get('epoch', 'N/A')}"):
                if available_epochs:
                    selected_epoch = st.selectbox(
                        "Época para avaliar",
                        available_epochs,
                        key=f"epoch_{model_name}",
                    )

                    if st.button(
                        "Avaliar modelo", key=f"eval_{model_name}", type="primary"
                    ):
                        with st.spinner(f"Avaliando {model_name}..."):
                            try:
                                result = evaluate_model(model_name, selected_epoch)
                                if "error" not in result:
                                    acc = result.get("metrics", {}).get("accuracy", 0)
                                    st.success(
                                        f"Avaliação concluída! Acurácia: {acc:.1f}%"
                                    )
                                    st.rerun()
                                else:
                                    st.error(
                                        f"Erro: {result.get('error', 'Erro desconhecido')}"
                                    )
                            except Exception as e:
                                st.error(f"Erro ao avaliar: {e}")
                else:
                    st.warning("Nenhuma época disponível para este modelo")

    if not models_with_results:
        return

    # Detailed view
    st.divider()
    st.header("Detalhes")

    selected_model = st.selectbox(
        "Selecione um modelo",
        list(models_with_results.keys()),
        format_func=lambda x: f"{x} ({models_with_results[x]['accuracy']:.1f}%)",
    )

    if selected_model:
        data = models_data[selected_model]

        st.metric("Acurácia", f"{data['accuracy']:.1f}%")
        st.metric("Limiar Sugerido", f"{data['threshold']:.2f}")

        with st.expander("Estatísticas detalhadas"):
            st.text(f"Saudáveis: {data['healthy_count']} imagens")
            st.text(f"  Média: {data['healthy_mean']:.3f}")
            st.text(f"  Desvio: {data['healthy_std']:.3f}")
            st.text(f"Doentes: {data['diseased_count']} imagens")
            st.text(f"  Média: {data['diseased_mean']:.3f}")
            st.text(f"  Desvio: {data['diseased_std']:.3f}")

            # Distribution chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(
                data["healthy_scores"],
                bins=20,
                alpha=0.6,
                label="Saudáveis",
                color="#2E7D32",
            )
            ax.hist(
                data["diseased_scores"],
                bins=20,
                alpha=0.6,
                label="Doentes",
                color="#C62828",
            )
            ax.axvline(
                data["threshold"],
                color="black",
                linestyle="--",
                label=f'Limiar ({data["threshold"]:.2f})',
            )
            ax.set_xlabel("Score CIEDE2000")
            ax.set_ylabel("Frequência")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # Classification metrics
            threshold = data["threshold"]
            tp = sum(1 for s in data["diseased_scores"] if s > threshold)
            fn = sum(1 for s in data["diseased_scores"] if s <= threshold)
            tn = sum(1 for s in data["healthy_scores"] if s <= threshold)
            fp = sum(1 for s in data["healthy_scores"] if s > threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            st.text(f"Precisão: {precision*100:.1f}%")
            st.text(f"Recall: {recall*100:.1f}%")
            st.text(f"F1-Score: {f1*100:.1f}%")
