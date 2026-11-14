"""AccessAble AI entry point.

Phase 0: placeholder so the project layout is complete and the run command
is documented. The real Streamlit app shell (navigation, dashboard, image
upload workflow) is built in Phase 1.
"""

import streamlit as st

st.set_page_config(page_title="AccessAble AI", page_icon="Accessible AI", layout="wide")

st.title("AccessAble AI")
st.markdown(
    "**Educational prototype only.** AI may miss or misidentify objects. "
    "Do not rely on this app for safe mobility or obstacle avoidance."
)
st.info(
    "This is the Phase 0 scaffold. The full app shell, Scene Assistant, "
    "Read Text, Navigation Demo, and Settings pages arrive in Phase 1."
)
