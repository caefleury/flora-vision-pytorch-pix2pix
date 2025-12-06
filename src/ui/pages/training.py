import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from io import BytesIO

from ...training.manager import TrainingManager, get_training_manager
from ...training.config import TrainingConfig, TrainingProgress
from ...training.presets import get_training_presets, get_recommended_presets


def render():

    st.title("Treinamento de Modelo")
    st.caption("Treine um novo modelo de detecção com suas próprias imagens")

    manager = get_training_manager()

    if "training_started" not in st.session_state:
        st.session_state.training_started = False
    if "healthy_files" not in st.session_state:
        st.session_state.healthy_files = []
    if "diseased_files" not in st.session_state:
        st.session_state.diseased_files = []
    if "training_log" not in st.session_state:
        st.session_state.training_log = []

    is_training = manager.is_training()
    progress = manager.get_progress()

    if is_training or progress.status in ["evaluating", "completed"]:
        _render_training_progress(manager, progress)
    else:
        _render_training_setup(manager)


def _render_training_setup(manager):
    if "config_values" not in st.session_state:
        st.session_state.config_values = {}

    st.header("1. Upload das Imagens")

    st.subheader("Folhas Saudáveis (Treino)")
    healthy_files = st.file_uploader(
        "Selecione imagens de folhas saudáveis",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="healthy_upload",
        help="Usadas para treinar o modelo. Recomendado: 50+ imagens.",
    )
    if healthy_files:
        st.caption(f"{len(healthy_files)} imagens carregadas")

    st.subheader("Folhas Doentes (Teste)")
    diseased_files = st.file_uploader(
        "Selecione imagens de folhas doentes",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="diseased_upload",
        help="Usadas para avaliar o modelo. Recomendado: 20+ imagens.",
    )
    if diseased_files:
        st.caption(f"{len(diseased_files)} imagens carregadas")

    # Step 2: Configuration
    st.header("2. Configurações")

    num_healthy = len(healthy_files) if healthy_files else 0

    if num_healthy > 0:
        recommendations = get_recommended_presets(num_healthy)
        category = recommendations.get("category", "pequeno")
        presets = recommendations.get("presets", {})

        category_names = {
            "muito_pequeno": "Muito Pequeno",
            "pequeno": "Pequeno",
            "medio": "Médio",
            "grande": "Grande",
            "muito_grande": "Muito Grande",
        }
        name = category_names.get(category, category)
        st.caption(f"Dataset: {name} ({num_healthy} imagens)")

        # Preset selection
        if presets:
            preset_options = list(presets.keys())
            selected_preset = st.selectbox(
                "Preset de configuração",
                preset_options,
                index=(
                    preset_options.index("recomendado")
                    if "recomendado" in preset_options
                    else 0
                ),
            )

            if selected_preset:
                flags = presets[selected_preset].get("flags", {})
                st.session_state.config_values = flags.copy()

    # Mode selection
    config_mode = st.radio(
        "Modo de configuração",
        ["Simples", "Avançado"],
        horizontal=True,
        help="Modo simples: configurações pré-definidas. Modo avançado: controle total.",
    )

    if config_mode == "Simples":
        config = _render_simple_config()
    else:
        config = _render_advanced_config()

    # Step 3: Start Training
    st.header("3. Iniciar Treinamento")

    # Validation
    can_train = True
    if not healthy_files or len(healthy_files) < 10:
        can_train = False
        st.warning("Carregue pelo menos 10 imagens de folhas saudáveis")
    if not diseased_files or len(diseased_files) < 5:
        can_train = False
        st.warning("Carregue pelo menos 5 imagens de folhas doentes")

    # Training summary
    if can_train:
        total_epochs = config.n_epochs + config.n_epochs_decay
        st.caption(
            f"Modelo: {config.name} | Épocas: {total_epochs} | "
            f"Gerador: {config.netG} | LR: {config.lr}"
        )

        if st.button("Iniciar Treinamento", type="primary", use_container_width=True):
            healthy_data = [(f.name, BytesIO(f.read())) for f in healthy_files]
            diseased_data = [(f.name, BytesIO(f.read())) for f in diseased_files]

            for f in healthy_files:
                f.seek(0)
            for f in diseased_files:
                f.seek(0)

            success = manager.start_training(healthy_data, diseased_data, config)

            if success:
                st.session_state.training_started = True
                st.rerun()
            else:
                st.error("Erro ao iniciar treinamento.")
    else:
        st.button(
            "Iniciar Treinamento",
            type="primary",
            use_container_width=True,
            disabled=True,
        )


def _render_simple_config():

    presets = get_training_presets()

    preset_name = st.selectbox(
        "Preset de treinamento",
        list(presets.keys()),
        index=1,
        format_func=lambda x: {
            "rápido": "Rápido (50 épocas)",
            "padrão": "Padrão (100 épocas)",
            "completo": "Completo (200 épocas)",
        }.get(x, x),
    )

    model_name = st.text_input(
        "Nome do modelo",
        value=f"folhas_{preset_name}",
    )

    config = presets[preset_name]
    config.name = model_name.replace(" ", "_").lower()

    return config


def _get_index_safe(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default


def _render_advanced_config():

    cv = st.session_state.get("config_values", {})

    st.subheader("Configurações Básicas")

    model_name = st.text_input("Nome do modelo", value="folhas_custom")

    n_epochs = st.slider(
        "Épocas de treino",
        min_value=10,
        max_value=200,
        value=cv.get("n_epochs", 50),
        step=10,
    )

    n_epochs_decay = st.slider(
        "Épocas de decay",
        min_value=0,
        max_value=200,
        value=cv.get("n_epochs_decay", 50),
        step=10,
    )

    st.subheader("Hiperparâmetros")

    col1, col2, col3, col4 = st.columns(4)

    batch_options = [1, 2, 4, 8, 16]
    lr_options = [0.00005, 0.0001, 0.0002, 0.0005, 0.001]
    lr_policy_options = ["linear", "step", "plateau", "cosine"]

    with col1:
        batch_size = st.selectbox(
            "Batch size",
            batch_options,
            index=_get_index_safe(batch_options, cv.get("batch_size", 1), 0),
            help="Tamanho do batch",
        )

    with col2:
        lr = st.select_slider(
            "Learning rate",
            options=lr_options,
            value=(
                cv.get("lr", 0.0002) if cv.get("lr", 0.0002) in lr_options else 0.0002
            ),
            format_func=lambda x: f"{x:.5f}",
        )

    with col3:
        beta1 = st.slider(
            "Beta1 (Adam)",
            min_value=0.0,
            max_value=0.99,
            value=cv.get("beta1", 0.5),
            step=0.05,
        )

    with col4:
        lr_policy = st.selectbox(
            "LR Policy",
            lr_policy_options,
            index=_get_index_safe(lr_policy_options, cv.get("lr_policy", "linear"), 0),
        )

    st.subheader("Arquitetura de Rede")

    col1, col2, col3 = st.columns(3)

    netG_options = ["unet_256", "unet_128", "resnet_9blocks", "resnet_6blocks"]
    netD_options = ["basic", "n_layers", "pixel"]
    norm_options = ["batch", "instance", "none"]

    with col1:
        netG = st.selectbox(
            "Gerador (netG)",
            netG_options,
            index=_get_index_safe(netG_options, cv.get("netG", "unet_256"), 0),
        )

    with col2:
        netD = st.selectbox(
            "Discriminador (netD)",
            netD_options,
            index=_get_index_safe(netD_options, cv.get("netD", "basic"), 0),
        )

    with col3:
        norm = st.selectbox(
            "Normalização",
            norm_options,
            index=_get_index_safe(norm_options, cv.get("norm", "batch"), 0),
        )

    col1, col2, col3 = st.columns(3)

    ngf_options = [32, 64, 128]
    ndf_options = [32, 64, 128]

    with col1:
        ngf = st.selectbox(
            "Filtros gerador (ngf)",
            ngf_options,
            index=_get_index_safe(ngf_options, cv.get("ngf", 64), 1),
        )

    with col2:
        ndf = st.selectbox(
            "Filtros discriminador (ndf)",
            ndf_options,
            index=_get_index_safe(ndf_options, cv.get("ndf", 64), 1),
        )

    with col3:
        n_layers_D = st.slider("Camadas discriminador", 1, 5, cv.get("n_layers_D", 3))

    st.subheader("Data Augmentation")

    col1, col2, col3, col4 = st.columns(4)

    load_size_options = [256, 286, 320]
    crop_size_options = [128, 256]
    preprocess_options = ["resize_and_crop", "crop", "scale_width", "none"]

    with col1:
        load_size = st.selectbox(
            "Load size",
            load_size_options,
            index=_get_index_safe(load_size_options, cv.get("load_size", 286), 1),
        )

    with col2:
        crop_size = st.selectbox(
            "Crop size",
            crop_size_options,
            index=_get_index_safe(crop_size_options, cv.get("crop_size", 256), 1),
        )

    with col3:
        preprocess = st.selectbox(
            "Preprocess",
            preprocess_options,
            index=_get_index_safe(
                preprocess_options, cv.get("preprocess", "resize_and_crop"), 0
            ),
        )

    with col4:
        no_flip = st.checkbox("Desabilitar flip", value=cv.get("no_flip", False))

    st.subheader("Salvamento")

    col1, col2 = st.columns(2)

    gan_mode_options = ["lsgan", "vanilla", "wgangp"]

    with col1:
        save_epoch_freq = st.slider(
            "Salvar a cada N épocas",
            min_value=5,
            max_value=50,
            value=cv.get("save_epoch_freq", 10),
            step=5,
        )

    with col2:
        gan_mode = st.selectbox(
            "GAN Mode",
            gan_mode_options,
            index=_get_index_safe(gan_mode_options, cv.get("gan_mode", "lsgan"), 0),
        )

    return TrainingConfig(
        name=model_name.replace(" ", "_").lower(),
        n_epochs=n_epochs,
        n_epochs_decay=n_epochs_decay,
        batch_size=batch_size,
        lr=lr,
        beta1=beta1,
        lr_policy=lr_policy,
        netG=netG,
        netD=netD,
        ngf=ngf,
        ndf=ndf,
        n_layers_D=n_layers_D,
        norm=norm,
        load_size=load_size,
        crop_size=crop_size,
        preprocess=preprocess,
        no_flip=no_flip,
        gan_mode=gan_mode,
        save_epoch_freq=save_epoch_freq,
    )


def _render_training_progress(manager, progress):

    status = progress.status

    status_texts = {
        "preparing": "Preparando dados...",
        "training": "Treinando modelo...",
        "evaluating": "Avaliando modelo...",
        "completed": "Treinamento concluído!",
        "error": "Erro no treinamento",
    }

    text = status_texts.get(status, status)
    st.header(text)

    if status == "error":
        st.error(f"Erro: {progress.error}")
        if st.button("Reiniciar", type="primary"):
            manager.reset()
            st.session_state.training_started = False
            st.session_state.training_log = []
            st.session_state.config_values = {}
            st.session_state.selected_preset = None
            st.rerun()
        return

    if status == "completed":
        _render_training_results(progress)
        if st.button("Novo Treinamento", type="primary"):
            manager.reset()
            st.session_state.training_started = False
            st.session_state.training_log = []
            st.session_state.config_values = {}
            st.session_state.selected_preset = None
            st.rerun()
        return

    # Progress bars
    st.subheader("Progresso")

    if progress.total_epochs > 0:
        epoch_progress = progress.current_epoch / progress.total_epochs
        st.progress(
            epoch_progress,
            text=f"Época {progress.current_epoch} / {progress.total_epochs}",
        )

    # Time elapsed
    if progress.elapsed_time > 0:
        elapsed_min = int(progress.elapsed_time // 60)
        elapsed_sec = int(progress.elapsed_time % 60)

        if progress.current_epoch > 0 and progress.total_epochs > 0:
            remaining = (progress.elapsed_time / progress.current_epoch) * (
                progress.total_epochs - progress.current_epoch
            )
            remaining_min = int(remaining // 60)
            remaining_sec = int(remaining % 60)
            time_text = f"Decorrido: {elapsed_min}m {elapsed_sec}s | Restante: {remaining_min}m {remaining_sec}s"
        else:
            time_text = f"Decorrido: {elapsed_min}m {elapsed_sec}s"

        st.markdown(time_text)

    # Current losses
    if progress.losses:
        st.subheader("Perdas Atuais")
        cols = st.columns(len(progress.losses))
        for i, (name, value) in enumerate(progress.losses.items()):
            with cols[i]:
                st.metric(name, f"{value:.4f}")

    # Message
    if progress.message:
        st.info(progress.message)

    # Training log
    new_lines = manager.get_output_lines()
    if new_lines:
        st.session_state.training_log.extend(new_lines)
        st.session_state.training_log = st.session_state.training_log[-100:]

    with st.expander("Log de Treinamento"):
        log_text = "".join(st.session_state.training_log[-50:])
        st.code(log_text, language="text")

    if st.button("Cancelar Treinamento", type="secondary"):
        manager.stop_training()
        st.warning("Cancelando treinamento...")

    time.sleep(2)
    st.rerun()


def _render_training_results(progress):

    results = progress.evaluation_results

    if not results or "error" in results:
        error_msg = results.get("error", "Erro desconhecido") if results else "Sem resultados"
        st.warning(f"Não foi possível avaliar o modelo automaticamente: {error_msg}")
        return

    st.success("Treinamento concluído com sucesso!")

    if progress.elapsed_time > 0:
        total_min = int(progress.elapsed_time // 60)
        total_sec = int(progress.elapsed_time % 60)
        st.caption(f"Tempo total: {total_min}m {total_sec}s")

    st.subheader("Resultados da Avaliação")

    metrics = results.get("metrics", {})
    healthy_results = results.get("healthy", [])
    diseased_results = results.get("diseased", [])

    # Calculate metrics if not already present
    accuracy = metrics.get("accuracy")
    threshold = metrics.get("suggested_threshold")
    total_images = metrics.get("total_test_images", len(healthy_results) + len(diseased_results))
    
    # Calculate scores for display
    healthy_scores = [r["score"] for r in healthy_results] if healthy_results else []
    diseased_scores = [r["score"] for r in diseased_results] if diseased_results else []

    # Display main metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if accuracy is not None:
            st.metric("Acurácia", f"{accuracy:.1f}%")
        elif healthy_scores and diseased_scores:
            # Calculate accuracy if we have the data
            healthy_mean = np.mean(healthy_scores)
            healthy_std = np.std(healthy_scores) if len(healthy_scores) > 1 else 0
            calc_threshold = healthy_mean + 2 * healthy_std
            correct = sum(1 for s in healthy_scores if s < calc_threshold) + sum(1 for s in diseased_scores if s >= calc_threshold)
            calc_accuracy = (correct / total_images) * 100 if total_images > 0 else 0
            st.metric("Acurácia", f"{calc_accuracy:.1f}%")
        else:
            st.metric("Acurácia", "N/A")
    
    with col2:
        if threshold is not None:
            st.metric("Limiar Sugerido", f"{threshold:.2f}")
        elif healthy_scores:
            healthy_mean = np.mean(healthy_scores)
            healthy_std = np.std(healthy_scores) if len(healthy_scores) > 1 else 0
            calc_threshold = healthy_mean + 2 * healthy_std
            st.metric("Limiar Sugerido", f"{calc_threshold:.2f}")
        else:
            st.metric("Limiar Sugerido", "N/A")
    
    with col3:
        st.metric("Imagens Testadas", total_images)

    # Details expander
    with st.expander("Detalhes", expanded=True):
        if healthy_scores:
            h_mean = metrics.get('healthy_mean', np.mean(healthy_scores))
            h_std = metrics.get('healthy_std', np.std(healthy_scores) if len(healthy_scores) > 1 else 0)
            st.text(f"Saudáveis - Média: {h_mean:.2f}, Desvio: {h_std:.2f}")
        else:
            st.text("Saudáveis - Sem dados")
            
        if diseased_scores:
            d_mean = metrics.get('diseased_mean', np.mean(diseased_scores))
            d_std = metrics.get('diseased_std', np.std(diseased_scores) if len(diseased_scores) > 1 else 0)
            st.text(f"Doentes - Média: {d_mean:.2f}, Desvio: {d_std:.2f}")
        else:
            st.text("Doentes - Sem dados")

        # Plot histogram if we have data
        if healthy_scores or diseased_scores:
            fig, ax = plt.subplots(figsize=(8, 4))

            if healthy_scores:
                ax.hist(
                    healthy_scores,
                    bins=min(20, len(healthy_scores)),
                    alpha=0.6,
                    label=f"Saudáveis ({len(healthy_scores)})",
                    color="#2E7D32",
                )
            
            if diseased_scores:
                ax.hist(
                    diseased_scores,
                    bins=min(20, len(diseased_scores)),
                    alpha=0.6,
                    label=f"Doentes ({len(diseased_scores)})",
                    color="#C62828",
                )

            # Draw threshold line
            display_threshold = threshold
            if display_threshold is None and healthy_scores:
                display_threshold = np.mean(healthy_scores) + 2 * np.std(healthy_scores)
            
            if display_threshold is not None:
                ax.axvline(
                    display_threshold,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"Limiar ({display_threshold:.2f})",
                )

            ax.set_xlabel("Score CIEDE2000")
            ax.set_ylabel("Frequência")
            ax.legend()
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Nenhum dado de score disponível para visualização")

    st.info("O modelo foi salvo e está disponível na página de análise.")
