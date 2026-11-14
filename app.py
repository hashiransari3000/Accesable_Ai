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

from modules import object_detection as od
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
SESSION_RESULT = "detection_result"      # od.DetectionResult
SESSION_RESULT_SRC = "detection_source"  # image bytes the result belongs to
SESSION_RESULT_CONF = "detection_conf"   # threshold used for that result


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
        st.metric(
            "Detection model",
            "yolo11n (cached)" if od.is_model_loaded() else "loads on first use",
            help="Nano YOLO on CPU; first use downloads ~5 MB once, then offline.",
        )

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
    st.session_state.pop(SESSION_RESULT, None)
    st.session_state.pop(SESSION_RESULT_SRC, None)
    st.session_state.pop(SESSION_RESULT_CONF, None)
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

    # Stale-result guard: if the image changed since the last analysis,
    # old results must not be presented as current (reduce memory load).
    stale = (
        st.session_state.get(SESSION_RESULT) is not None
        and st.session_state.get(SESSION_RESULT_SRC) != image_bytes
    )

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

    if stale:
        components.status_banner(
            "warning",
            "The image changed since the last analysis. Press "
            "Analyze image again to refresh the results.",
        )

    # -- Analyze (real local YOLO inference, Phase 2) -------------------------
    st.divider()
    conf_threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.40,
        step=0.05,
        format="%.2f",
        help=(
            "Minimum model score for a detection to be shown. Confidence is "
            "a model score, not a probability of safety."
        ),
    )

    if st.button("Analyze image", type="primary"):
        first_run = not od.is_model_loaded()
        hint = (
            "Loading model (one-time ~5 MB download)..." if first_run
            else "Detecting objects locally..."
        )
        with st.spinner(hint):
            result = od.analyze_image(image_bytes, confidence_threshold=conf_threshold)
        st.session_state[SESSION_RESULT] = result
        st.session_state[SESSION_RESULT_SRC] = image_bytes
        st.session_state[SESSION_RESULT_CONF] = conf_threshold
        st.rerun()

    _render_detection_results(image_bytes)


def _render_detection_results(current_image_bytes: bytes) -> None:
    """Render the stored analysis result with all required states."""
    result: od.DetectionResult | None = st.session_state.get(SESSION_RESULT)
    if result is None or st.session_state.get(SESSION_RESULT_SRC) != current_image_bytes:
        return  # no result yet, or results belong to a previous image

    conf_used = st.session_state.get(SESSION_RESULT_CONF, od.DEFAULT_CONFIDENCE)

    if result.error:
        components.status_banner("error", result.error)
        if not result.detections:
            return

    if result.detections:
        components.status_banner(
            "success",
            f"{len(result.detections)} object(s) detected "
            f"(threshold {conf_used:.2f}, inference {result.inference_ms} ms on CPU).",
        )
    else:
        components.status_banner(
            "info",
            f"No objects detected above the threshold {conf_used:.2f}. "
            "Try lowering the threshold or using a clearer image. Note: the "
            "model recognizes only its training classes (80 COCO classes).",
        )

    # Annotated preview with a hide-overlay option (avoid clutter, spec 3.1)
    if result.annotated_png:
        show_boxes = st.checkbox(
            "Show bounding boxes on preview", value=True,
            help="Uncheck to view the original image instead of the overlay.",
        )
        if show_boxes:
            st.image(
                Image.open(io.BytesIO(result.annotated_png)),
                caption="Detected objects in this image",
                use_container_width=True,
            )

    # Structured detection list
    if result.detections:
        st.subheader("Detected objects")
        rows = [
            {
                "Object": d.label.replace("_", " "),
                "Confidence": f"{d.confidence * 100:.0f}%",
                "Location in image (x1, y1, x2, y2)": str(d.bbox),
            }
            for d in result.detections
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Confidence is the model's score for this image region - not a "
            "probability of safety. Descriptions refer to the uploaded image "
            "only, not your physical surroundings."
        )
    else:
        st.caption(
            "Tip: lower the threshold above and analyze again. The pretrained "
            "model only knows its training classes, so valid images can "
            "legitimately produce no detections."
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
    current_conf = st.session_state.get("conf_threshold", od.DEFAULT_CONFIDENCE)
    st.caption(
        f"Detection confidence threshold: currently **{current_conf:.2f}** - "
        "change it with the slider on the Scene Assistant page."
    )

    st.subheader("Model / device status")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            "- **Device:** CPU only (GPU optional, not required)\n"
            "- **Detection model:** yolo11n nano (Ultralytics, AGPL-3.0), "
            "cached locally after first use\n"
            "- **OCR model:** not loaded yet (Phase 4)"
        )
    with d2:
        st.markdown(
            "- **Speech:** eSpeak NG detected on this system\n"
            "- **Detection model:** yolo11n nano, CPU inference "
            "(one-time ~5 MB download, then offline)\n"
            "- **OCR model:** not loaded yet (Phase 4)\n"
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
