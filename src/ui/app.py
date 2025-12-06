import streamlit as st

from .styles import apply_custom_styles
from .pages import analysis, training, comparison, about
from ..utils.paths import ASSETS_DIR


def configure_page():
    icon_path = ASSETS_DIR / "leaf_icon.png"

    st.set_page_config(
        page_title="Sistema de Detecção de Doenças em Folhas",
        page_icon=str(icon_path) if icon_path.exists() else "🍃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <h1 style="margin: 0; color: #4CAF50; border: none; font-size: 2rem;">🍃 FloraVision</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("### 📂 Ferramentas")
        
        pages = {
            "🔬 Análise de Folhas": analysis.render,
            "⚙️ Treinamento": training.render,
            "📊 Modelos": comparison.render,
            "ℹ️ Sobre": about.render,
        }
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = list(pages.keys())[0]
        
        for page_name in pages.keys():
            is_selected = st.session_state.current_page == page_name
            if st.button(
                page_name,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.current_page = page_name
                st.rerun()
        
    return pages[st.session_state.current_page]


def main():
    configure_page()
    page_renderer = render_sidebar()
    page_renderer()


if __name__ == "__main__":
    main()
