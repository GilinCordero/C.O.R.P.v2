"""Session state management."""
import streamlit as st


def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None


def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def login(username: str):
    st.session_state.authenticated = True
    st.session_state.username = username


def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
