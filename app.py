"""AccessAble AI - application entry point.

Thin UI shell. Business logic lives in modules/; shared visual pieces in
ui/. Routing is a simple sidebar radio (predictable, keyboard-friendly).

Pages:
  Dashboard            - quick start, feature cards, readiness, notices
  Scene Assistant      - image upload workflow (detection lands Phase 2)
  Read Text            - OCR (lands Phase 4)
  Indoor Navigation    - BFS / A* demo (lands Phase 6)
  Settings & Help      - preferences, privacy, limitations
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from modules.image_validation import DEFAULT_MAX_BYTES, validate_upload
from ui import components, styles

st.set_page_config(
    page_title="AccessAble AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)
styles.inject_global_css()

PAGES = [
    "Dashboard",
    "Scene Assistant",
    "Read Text",
    "Indoor Navigation Demo",
    "Settings & Help",
]

SAMPLE_ROOM = "assets/sample_images/sample_room.png"
SAMPLE_SIGN = "assets/sample_images/sample_sign.png"

SESSION_IMAGE = "scene_image_bytes"
SESSION_NAME = "scene_image_name"
SESSION_CLEAR = "scene_confirm_clear"


def go(page: str) -> None:
    """Programmatic navigation helper (used by dashboard shortcuts)."""
    st.session_state["page"] = page
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## AccessAble AI")
        st.caption("Offline-first visual assistant prototype")
        page = st.radio(
            "Navigation",
            PAGES,
            key="page",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("🔒 All processing stays on this machine.")
    return page


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    components.page_header(
        "Welcome to AccessAble AI",
        "Understand a scene, read visible text, and explore indoor routes - locally and offline.",
    )
    components.safety_notice()

    st.subheader("Quick start")
    components.quick_start_steps()

    st.subheader("What this app can do")
    c1, c2 = st.columns(2)
    with c1:
        components.feature_card(
            "🖼️",
            "Scene Assistant",
            "Upload an image to detect common objects with a small pretrained "
            "YOLO model running on your CPU, with confidence values and "
            "relative-position descriptions.",
        )
        components.feature_card(
            "🔤",
            "Read Text",
            "Extract printed text from an image with local OCR and have it "
            "spoken aloud. Loads only when you request it.",
        )
    with c2:
        components.feature_card(
            "🔊",
            "Voice feedback",
            "Concise spoken summaries using offline text-to-speech, with "
            "speak, stop, replay, and rate controls. A visual text version "
            "is always provided.",
        )
        components.feature_card(
            "🧭",
            "Indoor Navigation Demo",
            "A clearly fictional sample map used to demonstrate BFS and A* "
            "route planning. Simulation only - never live guidance.",
        )

    st.subheader("System readiness")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Python", "3.14", help="Local runtime")
    with r2:
        st.metric("Speech engine", "eSpeak NG", help="Offline TTS backend")
    with r3:
        st.metric("Detection model", "not loaded", help="Loads on first use in Phase 2")

    components.privacy_notice()
    if st.button("Open Scene Assistant", type="primary"):
        go("Scene Assistant")


# ---------------------------------------------------------------------------
# Scene Assistant (Phase 1: upload workflow; Phase 2 adds real detection)
# ---------------------------------------------------------------------------
def _load_sample(path: str) -> None:
    with open(path, "rb") as fh:
        st.session_state[SESSION_IMAGE] = fh.read()
    st.session_state[SESSION_NAME] = path.rsplit("/", 1)[-1]
    st.session_state.pop(SESSION_CLEAR, None)
    st.rerun()


def _clear_image() -> None:
    st.session_state.pop(SESSION_IMAGE, None)
    st.session_state.pop(SESSION_NAME, None)
    st.session_state.pop(SESSION_CLEAR, None)
    st.rerun()


def render_scene_assistant() -> None:
    components.page_header(
        "Scene Assistant",
        "Upload an image to detect common objects and get a spoken summary. "
        "Detection runs locally on your machine.",
    )
    components.privacy_notice()

    # -- Input: upload ------------------------------------------------------
    uploaded = st.file_uploader(
        "Upload an image (JPG or PNG, up to 10 MB)",
        type=["jpg", "jpeg", "png"],
        key="scene_uploader",
    )

    if uploaded is not None:
        # Cheap checks first: extension and size (error prevention)
        error = validate_upload(uploaded.name, uploaded.size)
        if error is None:
            data = uploaded.getvalue()
            error = validate_upload(uploaded.name, uploaded.size, data)
        if error:
            components.status_banner("error", error)
        else:
            st.session_state[SESSION_IMAGE] = data
            st.session_state[SESSION_NAME] = uploaded.name

    # -- Sample images (universal usability: users without files at hand) ---
    st.markdown("**No image at hand? Try a sample:**")
    s1, s2, _ = st.columns([1, 1, 3])
    with s1:
        st.button("Sample: room", on_click=_load_sample, args=(SAMPLE_ROOM,))
    with s2:
        st.button("Sample: sign", on_click=_load_sample, args=(SAMPLE_SIGN,))

    image_bytes = st.session_state.get(SESSION_IMAGE)
    image_name = st.session_state.get(SESSION_NAME)

    # -- Preview + reset (easy reversal) ------------------------------------
    if image_bytes is None:
        components.status_banner(
            "info", "No image selected yet. Upload a file or pick a sample above."
        )
        return

    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**Selected image:** `{image_name}`")
            st.image(
                Image.open(io.BytesIO(image_bytes)),
                use_container_width=True,
            )
        with right:
            st.markdown("#### Manage")
            st.caption("Clearing removes the image from this page.")
            if st.session_state.get(SESSION_CLEAR):
                st.caption("Are you sure? Press confirm to clear.")
                if st.button("Confirm clear", type="primary"):
                    _clear_image()
            elif st.button("Clear image"):
                st.session_state[SESSION_CLEAR] = True
                st.rerun()

    # -- Analyze (placeholder until Phase 2) ---------------------------------
    st.divider()
    if st.button("Analyze image", type="primary"):
        with st.spinner("Processing image..."):
            import time

            time.sleep(1.0)  # stand-in for inference latency until Phase 2

        components.status_banner(
            "info",
            "Object detection is not connected yet. This placeholder keeps "
            "the workflow honest until Phase 2 adds the real YOLO model.",
        )
        components.placeholder_panel(
            "Detections with labels, confidence values, and relative positions",
            "Phase 2",
        )
        components.placeholder_panel(
            "Spoken summary with speak / stop / replay controls", "Phase 3"
        )


# ---------------------------------------------------------------------------
# Read Text (Phase 4)
# ---------------------------------------------------------------------------
def render_read_text() -> None:
    components.page_header(
        "Read Text",
        "Extract printed text from an image with local OCR and have it read aloud.",
    )
    components.privacy_notice()
    components.placeholder_panel("Local OCR text extraction and read-aloud", "Phase 4")
    components.status_banner(
        "info", "OCR models are not loaded in this build. Nothing to configure yet."
    )


# ---------------------------------------------------------------------------
# Indoor Navigation Demo (Phase 6)
# ---------------------------------------------------------------------------
def render_navigation() -> None:
    components.page_header(
        "Indoor Navigation Demo",
        "Explore BFS and A* route planning on a fictional sample building map.",
    )
    components.safety_notice()  # mandatory on the navigation page
    components.status_banner(
        "warning",
        "Simulation only. This page will use a fixed sample map; it does not "
        "know your live location and cannot detect real obstacles.",
    )
    components.placeholder_panel(
        "Sample map, start/destination selectors, BFS vs A* comparison", "Phase 6"
    )


# ---------------------------------------------------------------------------
# Settings & Help
# ---------------------------------------------------------------------------
def render_settings() -> None:
    components.page_header(
        "Settings & Help",
        "Preferences, system status, privacy explanation, and known limitations.",
    )

    st.subheader("Preferences")
    st.slider(
        "Speech rate",
        min_value=100,
        max_value=220,
        value=160,
        step=10,
        format="%d words/min",
        help="Takes effect when voice feedback arrives in Phase 3.",
        disabled=True,
    )
    st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.40,
        step=0.05,
        format="%.2f",
        help="Minimum model score for a detection to be shown. Available in Phase 2.",
        disabled=True,
    )

    st.subheader("Model / device status")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            "- **Device:** CPU only (GPU optional, not required)\n"
            "- **Detection model:** not loaded yet (Phase 2)\n"
            "- **OCR model:** not loaded yet (Phase 4)"
        )
    with d2:
        st.markdown(
            "- **Speech:** eSpeak NG detected on this system\n"
            "- **Network use:** one-time model download only\n"
            "- **Data retention:** none"
        )

    st.subheader("Privacy")
    st.markdown(
        "Every image you upload is processed locally. The app never uploads, "
        "stores, or transmits your images, OCR text, or any other data. There "
        "are no accounts, analytics, or telemetry."
    )

    st.subheader("Known limitations")
    st.markdown(
        "- Educational prototype only; not a mobility or safety device.\n"
        "- Detection recognizes only classes in the model's training labels.\n"
        "- Image position is not physical distance; depth is out of scope.\n"
        "- OCR supports English only (until other languages are tested).\n"
        "- The navigation demo uses a fixed fictional map."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main() -> None:
    page = render_sidebar()
    if page == "Dashboard":
        render_dashboard()
    elif page == "Scene Assistant":
        render_scene_assistant()
    elif page == "Read Text":
        render_read_text()
    elif page == "Indoor Navigation Demo":
        render_navigation()
    else:
        render_settings()


if __name__ == "__main__":
    main()
