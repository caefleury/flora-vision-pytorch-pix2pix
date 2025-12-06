import streamlit as st

CUSTOM_CSS = """
<style>
    h1 {
        color: #1a472a;
        border-bottom: 2px solid #1a472a;
        padding-bottom: 0.5rem;
    }
    h2 {
        color: #2d5a3d;
        margin-top: 1.5rem;
    }
    h3 {
        color: #3d6b4d;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: #e8f5e9;
        color: #1a472a;
        border: 2px solid #2d5a3d;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: #2d5a3d;
        color: white;
        transform: translateX(5px);
        border-color: #2d5a3d;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #1a472a;
        color: white;
        border: 2px solid #1a472a;
        box-shadow: 0 4px 12px rgba(26, 71, 42, 0.3);
    }
    
    .stMetric {
        background-color: transparent;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    
    [data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        background-color: transparent !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    .stProgress > div > div > div > div {
        background-color: #1a472a;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 500;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
</style>
"""


def apply_custom_styles():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
