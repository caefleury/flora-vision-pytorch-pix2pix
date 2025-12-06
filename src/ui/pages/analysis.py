import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

from ...core.detector import (
    LeafDiseaseDetector,
    get_available_models,
    get_available_epochs,
)
from ...visualization.heatmaps import create_heatmap_figure, create_simple_overlay
from ...visualization.gradcam import (
    ColorModelGradCAM,
    get_available_cam_methods,
    get_cam_method_description,
    apply_colormap,
    overlay_cam_on_image,
)


@st.cache_resource
def load_detector(model_name, epoch):
    return LeafDiseaseDetector(model_name, epoch)


def render():

    st.title("Análise de Folhas")
    st.caption("Detecte doenças em folhas através de análise de colorização")

    available_models = get_available_models()

    if not available_models:
        st.error("Nenhum modelo treinado encontrado.")
        st.info("Treine um modelo primeiro na aba de treinamento.")
        return

    with st.expander("Configuracoes do Modelo", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_model = st.selectbox(
                "Modelo",
                available_models,
                help="Modelo treinado para detecção",
            )

        with col2:
            available_epochs = get_available_epochs(selected_model)
            selected_epoch = st.selectbox(
                "Época",
                available_epochs,
                index=len(available_epochs) - 1 if available_epochs else 0,
            )

        with col3:
            threshold = st.slider(
                "Limiar de detecção",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Score CIEDE2000 acima deste valor indica doença",
            )

    st.header("Upload de Imagem")

    uploaded_file = st.file_uploader(
        "Selecione uma imagem de folha",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem carregada", use_container_width=True)

        if st.button("Analisar", type="primary", use_container_width=True):
            with st.spinner("Processando..."):
                try:
                    detector = load_detector(selected_model, selected_epoch)
                    result = detector.process_image(image)
                    result["threshold"] = threshold
                    result["is_diseased"] = result["score"] > threshold

                    st.session_state["result"] = result
                    st.session_state["model_info"] = {
                        "model": selected_model,
                        "epoch": selected_epoch,
                    }
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")
                    return

    if "result" in st.session_state:
        _render_results(threshold)


def _render_results(threshold):
    st.divider()
    st.header("Resultado")

    result = st.session_state["result"]
    model_info = st.session_state["model_info"]
    result["is_diseased"] = result["score"] > threshold

    if result["is_diseased"]:
        st.error("**Resultado: Folha com possivel doenca detectada**")
    else:
        st.success("**Resultado: Folha aparentemente saudavel**")

    st.metric(
        label="Score CIEDE2000",
        value=f"{result['score']:.2f}",
        delta=f"{result['score'] - threshold:+.2f} do limiar",
        delta_color="inverse" if result["is_diseased"] else "normal",
    )

    st.caption(f"Modelo: {model_info['model']} | Epoca: {model_info['epoch']}")

    st.subheader("Mapa de Anomalias")
    overlay = create_simple_overlay(result)
    st.image(
        overlay,
        caption="Regioes em vermelho indicam maior diferenca de cor",
        use_container_width=True,
    )

    with st.expander("Visualizacao detalhada"):
        fig = create_heatmap_figure(result)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Estatisticas")
        heatmap = result["heatmap"]
        stats_data = {
            "Media": f"{np.mean(heatmap):.2f}",
            "Maximo": f"{np.max(heatmap):.2f}",
            "Minimo": f"{np.min(heatmap):.2f}",
            "Desvio Padrao": f"{np.std(heatmap):.2f}",
        }
        for label, value in stats_data.items():
            st.text(f"{label}: {value}")

    with st.expander("Download"):
        fig = create_heatmap_figure(result)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        st.download_button(
            label="Baixar visualização completa",
            data=buf,
            file_name="resultado_analise.png",
            mime="image/png",
        )

    _render_gradcam_section(result, model_info)


def _render_gradcam_section(result, model_info):
    st.divider()
    st.header("Visualização Grad-CAM")
    st.caption(
        "Visualize quais regiões da rede neural são mais importantes para a colorização"
    )

    with st.expander("Configurar Grad-CAM", expanded=True):
        cam_methods = get_available_cam_methods()
        selected_cam = st.selectbox(
            "Método CAM",
            cam_methods,
            format_func=lambda x: x.upper(),
            help="Selecione o tipo de visualização CAM",
        )
        st.caption(get_cam_method_description(selected_cam))

        if st.button("Gerar Grad-CAM", type="primary"):
            with st.spinner("Gerando visualização Grad-CAM..."):
                try:
                    # Get detector and model
                    detector = load_detector(model_info["model"], model_info["epoch"])

                    # Initialize Grad-CAM
                    cam_generator = ColorModelGradCAM(
                        detector.model, method=selected_cam
                    )

                    # Get available layers
                    available_layers = cam_generator.get_available_layers()

                    if not available_layers:
                        st.warning(
                            "Não foi possível encontrar camadas para visualização."
                        )
                    else:
                        # Prepare input tensor
                        data = detector.preprocess_image(
                            Image.fromarray(result["original"])
                        )
                        input_tensor = data["A"].to(detector.opt.device)

                        # Select layers to visualize (first, middle, last)
                        n_layers = len(available_layers)
                        layer_indices = [0, n_layers // 2, n_layers - 1]
                        selected_layers = [
                            available_layers[i] for i in layer_indices if i < n_layers
                        ]

                        # Generate CAMs
                        st.subheader("Mapas de Ativação por Camada")

                        for layer_name in selected_layers:
                            try:
                                cam_heatmap = cam_generator.generate_cam(
                                    input_tensor, layer_name
                                )

                                # Create overlay
                                overlay = overlay_cam_on_image(
                                    result["generated"], cam_heatmap, alpha=0.5
                                )

                                # Display
                                st.text(f"Camada: {layer_name}")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(
                                        apply_colormap(cam_heatmap),
                                        caption="Mapa de calor",
                                        use_container_width=True,
                                    )
                                with col2:
                                    st.image(
                                        overlay,
                                        caption="Sobreposição",
                                        use_container_width=True,
                                    )

                            except Exception as e:
                                st.warning(f"Erro na camada {layer_name}: {e}")

                        st.success(
                            f"Grad-CAM ({selected_cam.upper()}) gerado com sucesso!"
                        )

                except Exception as e:
                    st.error(f"Erro ao gerar Grad-CAM: {e}")
                    import traceback

                    st.code(traceback.format_exc())
