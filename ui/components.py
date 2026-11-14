"""Reusable UI components for AccessAble AI.

Each component pairs colour with text (never colour alone) and keeps
essential information visible to reduce memory load.
"""

import streamlit as st


def page_header(title: str, description: str) -> None:
    """Consistent page title + one-line description on every screen."""
    st.title(title)
    st.caption(description)


def status_banner(kind: str, message: str) -> None:
    """Show a status message. kind: success | warning | error | info.

    Streamlit banners include their own icons/text, so state is never
    communicated by colour alone.
    """
    if kind == "success":
        st.success(message, icon="✅")
    elif kind == "warning":
        st.warning(message, icon="⚠️")
    elif kind == "error":
        st.error(message, icon="⛔")
    else:
        st.info(message, icon="ℹ️")


def safety_notice() -> None:
    """The mandatory visible disclaimer, shown on Dashboard + Navigation."""
    st.warning(
        "**Educational prototype only.** AI may miss or misidentify objects. "
        "Do not rely on this app for safe mobility or obstacle avoidance.",
        icon="⚠️",
    )


def privacy_notice() -> None:
    """Short, always-visible privacy statement."""
    st.caption(
        "🔒 Privacy: your image is processed locally on this machine. "
        "Nothing is uploaded, stored, or transmitted."
    )


def feature_card(icon: str, title: str, body: str) -> None:
    """Compact feature summary card for the dashboard grid."""
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**")
        st.markdown(body)


def quick_start_steps() -> None:
    """Visible instructions so new users never rely on memory."""
    st.markdown(
        "1. Open **Scene Assistant** in the sidebar.\n"
        "2. Upload an image (JPG or PNG) or pick a sample.\n"
        "3. Press **Analyze image** and read the results."
    )


def placeholder_panel(feature: str, phase: str) -> None:
    """Honest placeholder for features not yet implemented.

    We never pretend a feature exists before its phase lands.
    """
    with st.container(border=True):
        st.markdown(f"**{feature}**")
        st.caption(f"Under construction — arrives in {phase}.")
