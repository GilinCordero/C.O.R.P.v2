"""
C.O.R.P. v2 - Concrete Operations Real-Time Predictions
Entry point for the Streamlit app.
Hourly prediction with native LightGBM model.
"""
import sys
from pathlib import Path

# Add project root to path so 'app' is importable when running streamlit run app/app.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from auth_manager import verify_credentials, hash_password
from utils.cache_manager import init_session_state, is_authenticated, login, logout

st.set_page_config(
    page_title="C.O.R.P. v2",
    page_icon="app/assets/logo_GCC.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()


def show_login_page():
    """Render login form."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image("./app/assets/logo_GCC.png", width=200)
        st.title("C.O.R.P. v2")
        st.caption("Concrete Operations Real-Time Predictions")
        st.markdown("---")

        username = st.text_input("Usuario", value="gcc_corp_user")
        password = st.text_input("Contrasena", type="password")

        if st.button("Iniciar Sesion", use_container_width=True):
            if verify_credentials(username, password):
                login(username)
                st.success("Bienvenido!")
                st.rerun()
            else:
                st.error("Usuario o contrasena incorrectos.")


def show_main_app():
    """Render main app after login."""
    st.sidebar.image("./app/assets/logo_GCC.png", width=150)
    st.sidebar.title("C.O.R.P. v2")
    st.sidebar.markdown("Prediccion horaria de demanda")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navegacion",
        ["Prediccion de Demanda", "Gestion de Datos"],
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("Cerrar Sesion", use_container_width=True):
        logout()
        st.rerun()

    if page == "Prediccion de Demanda":
        from components.forecast_page import show_forecast_page
        show_forecast_page()
    elif page == "Gestion de Datos":
        from components.data_management_page import show_data_management_page
        show_data_management_page()


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------
if not is_authenticated():
    show_login_page()
else:
    show_main_app()
