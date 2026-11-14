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
import time

import streamlit as st
from PIL import Image

from modules import announcement_manager as am
from modules import camera as cam_mod
from modules import object_detection as od
from modules import text_recognition as tr
from modules import voice_assistant as vas
from modules.position_analysis import describe_position
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

OCR_SOURCE = "ocr_image_bytes"
OCR_NAME = "ocr_image_name"
OCR_CLEAR = "ocr_confirm_clear"
OCR_RESULT = "ocr_result"     # tr.TextRecognitionResult
OCR_RESULT_SRC = "ocr_source"

CAMERA = "camera_instance"        # cam_mod.CameraCapture
CAMERA_AUTO = "camera_auto_preview"
CAMERA_SNAPSHOT = "camera_snapshot_bytes"
CAMERA_RESULT = "camera_result"   # od.DetectionResult for the snapshot
LAST_PAGE = "last_active_page"


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

    # Camera mode (Phase 5) lives on the same page, after image analysis.
    render_camera_section()


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

    # Structured detection list, now with relative position (Phase 3)
    if result.detections:
        st.subheader("Detected objects")
        rows = [
            {
                "Object": d.label.replace("_", " "),
                "Confidence": f"{d.confidence * 100:.0f}%",
                "Position in image": describe_position(
                    d, result.image_size[0], result.image_size[1]
                ).replace("appears on the ", "").replace(" side of the image", ""),
                "Location in image (x1, y1, x2, y2)": str(d.bbox),
            }
            for d in result.detections
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Confidence is the model's score for this image region - not a "
            "probability of safety. Position describes the uploaded image "
            "only, never your physical surroundings or distances."
        )
    else:
        st.caption(
            "Tip: lower the threshold above and analyze again. The pretrained "
            "model only knows its training classes, so valid images can "
            "legitimately produce no detections."
        )

    # -- Spoken summary (Phase 3): text always visible, speech user-triggered
    _render_speech_controls(result)


def _render_speech_controls(
    result: od.DetectionResult, heading: str = "Spoken summary"
) -> None:
    """Concise spoken summary with speak / stop / replay / rate controls.

    Speech is never started automatically (internal locus of control);
    the text equivalent is always on screen (audio never the only channel).
    """
    if not result.detections:
        return

    summary = am.summarize(result.detections, result.image_size[0], result.image_size[1])

    st.subheader(heading)
    with st.container(border=True):
        st.markdown(f"**Text version:** {summary.summary}")

        tts_ok = vas.is_available()
        if not tts_ok:
            components.status_banner("warning", vas.setup_hint())

        st.caption("Speech rate")
        rate = st.slider(
            "Speech rate (words per minute)",
            min_value=100,
            max_value=220,
            value=vas.assistant.rate,
            step=10,
            key="speech_rate",
            label_visibility="collapsed",
            disabled=not tts_ok,
            help="Adjust before pressing Speak; applies to the next utterance.",
        )
        vas.assistant.set_rate(rate)

        b1, b2, b3 = st.columns(3)
        speak_pressed = b1.button(
            "Speak summary", type="primary", disabled=not tts_ok,
        )
        b2.button("Stop speech", disabled=not tts_ok, key="stop_speech")
        b3.button(
            "Replay speech",
            disabled=not tts_ok or not vas.assistant.has_last_text(),
            key="replay_speech",
        )

        if speak_pressed:
            if vas.assistant.speak(summary.summary):
                st.session_state["speech_status"] = "Speaking. Use Stop to interrupt."
            else:
                st.session_state["speech_status"] = "Speech could not start."
        if st.session_state.get("stop_speech"):
            vas.assistant.stop()
            st.session_state["speech_status"] = "Speech stopped."
        if st.session_state.get("replay_speech"):
            if vas.assistant.replay():
                st.session_state["speech_status"] = "Replaying last summary."
            else:
                st.session_state["speech_status"] = "Speech could not start."

        status = st.session_state.get("speech_status", "")
        if status:
            if vas.assistant.is_speaking():
                st.info(f"🔊 {status}", icon="🗣️")
            else:
                st.caption(status)
        elif tts_ok and not vas.assistant.is_speaking():
            st.caption("Not speaking. Press Speak summary to start.")


# ---------------------------------------------------------------------------
# Live camera mode (Phase 5) - deliberate activation, snapshot inference
# ---------------------------------------------------------------------------
def _stop_camera() -> None:
    """Stop the camera and drop its state. Always safe to call."""
    cam_inst = st.session_state.get(CAMERA)
    if cam_inst is not None:
        cam_inst.stop()
    st.session_state.pop(CAMERA, None)
    st.session_state.pop(CAMERA_SNAPSHOT, None)
    st.session_state.pop(CAMERA_RESULT, None)


def _encode_frame_png(frame) -> bytes | None:
    import cv2

    ok, buf = cv2.imencode(".png", frame)
    return buf.tobytes() if ok else None


def render_camera_section() -> None:
    st.divider()
    st.subheader("Live camera")
    st.caption(
        "🔒 Camera frames are processed locally in memory and are never "
        "saved, uploaded, or transmitted. The camera stays OFF until you "
        "press Start."
    )

    cam_inst = st.session_state.get(CAMERA)

    if not (cam_inst and cam_inst.is_active()):
        if cam_inst is not None and cam_inst.get_error():
            components.status_banner("error", cam_inst.get_error())
        st.button(
            "Start camera",
            key="start_camera",
            help="Deliberate action: nothing starts on page load.",
        )
        if st.session_state.get("start_camera"):
            new_cam = cam_mod.CameraCapture()
            error = new_cam.start()
            if error:
                components.status_banner("error", error)
            else:
                st.session_state[CAMERA] = new_cam
                st.rerun()
        return

    # -- Camera is active ------------------------------------------------------
    components.status_banner(
        "success",
        "Camera is active. Press Stop camera when you are done - it also "
        "stops automatically when you leave this page.",
    )
    if cam_inst.get_error():
        components.status_banner("error", cam_inst.get_error())
        _stop_camera()
        st.rerun()

    auto = st.checkbox(
        "Live preview (auto-refresh)",
        value=st.session_state.get(CAMERA_AUTO, False),
        help="Refreshes the preview every interval. Inference still runs "
        "only when you press Capture & analyze.",
        key="auto_preview",
    )
    st.session_state[CAMERA_AUTO] = auto
    if auto:
        interval = st.slider(
            "Preview refresh interval (seconds)", 0.5, 3.0, 1.0, 0.5,
            key="preview_interval",
        )

    frame = cam_inst.get_frame()
    if frame is None:
        st.caption("Waiting for the first frame...")
    else:
        st.image(
            frame[:, :, ::-1],  # BGR -> RGB for display
            caption="Live preview (uncaptured frames are not analyzed)",
            use_container_width=True,
        )

    b1, b2, _ = st.columns([1.4, 1.4, 2])
    capture_pressed = b1.button("Capture & analyze", type="primary")
    b2.button("Stop camera", key="stop_camera")

    if st.session_state.get("stop_camera"):
        _stop_camera()
        st.rerun()

    if capture_pressed:
        if frame is None:
            components.status_banner("warning", "No frame available yet.")
        else:
            snapshot = _encode_frame_png(frame)
            with st.spinner("Analyzing snapshot..."):
                result = od.analyze_image(
                    snapshot,
                    confidence_threshold=st.session_state.get("conf_threshold", od.DEFAULT_CONFIDENCE),
                )
            st.session_state[CAMERA_SNAPSHOT] = snapshot
            st.session_state[CAMERA_RESULT] = result

    _render_camera_results()

    # Throttled preview loop: capture rate (15 fps in the grabber thread)
    # stays decoupled from the UI refresh and from inference.
    if auto and cam_inst.is_active():
        time.sleep(st.session_state.get("preview_interval", 1.0))
        st.rerun()


def _render_camera_results() -> None:
    result: od.DetectionResult | None = st.session_state.get(CAMERA_RESULT)
    if result is None:
        return
    st.markdown("**Snapshot analysis** (from the moment you pressed Capture):")
    if result.error:
        components.status_banner("error", result.error)
        if not result.detections:
            return
    snapshot = st.session_state.get(CAMERA_SNAPSHOT)
    if result.annotated_png and snapshot:
        st.image(
            Image.open(io.BytesIO(result.annotated_png)),
            caption="Analyzed snapshot",
            use_container_width=True,
        )
    if result.detections:
        rows = [
            {
                "Object": d.label.replace("_", " "),
                "Confidence": f"{d.confidence * 100:.0f}%",
                "Position in image": describe_position(
                    d, result.image_size[0], result.image_size[1]
                ).replace("appears on the ", "").replace(" side of the image", ""),
            }
            for d in result.detections
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        components.status_banner(
            "info",
            "No objects detected in this snapshot. The model recognizes only "
            "its training classes (80 COCO classes).",
        )
    _render_speech_controls(result, heading="Spoken snapshot summary")


# ---------------------------------------------------------------------------
# Read Text (Phase 4: real local OCR)
# ---------------------------------------------------------------------------
def _load_ocr_sample(path: str) -> None:
    with open(path, "rb") as fh:
        st.session_state[OCR_SOURCE] = fh.read()
    st.session_state[OCR_NAME] = path.rsplit("/", 1)[-1]
    st.session_state.pop(OCR_CLEAR, None)
    st.rerun()


def _clear_ocr_image() -> None:
    for key in (OCR_SOURCE, OCR_NAME, OCR_CLEAR, OCR_RESULT, OCR_RESULT_SRC):
        st.session_state.pop(key, None)
    st.rerun()


def render_read_text() -> None:
    components.page_header(
        "Read Text",
        "Extract printed text from an image with local OCR and have it read aloud.",
    )
    components.privacy_notice()
    components.status_banner(
        "info",
        "English only for now. The OCR model loads on demand (one-time "
        "~85 MB download on first use), so this feature costs no memory "
        "until you use it.",
    )

    uploaded = st.file_uploader(
        "Upload an image with printed text (JPG or PNG, up to 10 MB)",
        type=["jpg", "jpeg", "png"],
        key="ocr_uploader",
    )
    if uploaded is not None:
        error = validate_upload(uploaded.name, uploaded.size)
        if error is None:
            data = uploaded.getvalue()
            error = validate_upload(uploaded.name, uploaded.size, data)
        if error:
            components.status_banner("error", error)
        else:
            st.session_state[OCR_SOURCE] = data
            st.session_state[OCR_NAME] = uploaded.name

    st.button("Sample: sign", on_click=_load_ocr_sample, args=(SAMPLE_SIGN,))

    image_bytes = st.session_state.get(OCR_SOURCE)
    image_name = st.session_state.get(OCR_NAME)
    if image_bytes is None:
        components.status_banner(
            "info", "No image selected yet. Upload a file or pick the sample above."
        )
        return

    stale = (
        st.session_state.get(OCR_RESULT) is not None
        and st.session_state.get(OCR_RESULT_SRC) != image_bytes
    )
    if stale:
        components.status_banner(
            "warning",
            "The image changed since the last extraction. Press "
            "Extract text again to refresh.",
        )

    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**Selected image:** `{image_name}`")
            st.image(Image.open(io.BytesIO(image_bytes)), use_container_width=True)
        with right:
            st.markdown("#### Manage")
            if st.session_state.get(OCR_CLEAR):
                st.caption("Are you sure? Press confirm to clear.")
                if st.button("Confirm clear", type="primary"):
                    _clear_ocr_image()
            elif st.button("Clear image"):
                st.session_state[OCR_CLEAR] = True
                st.rerun()

    st.divider()
    if st.button("Extract text", type="primary"):
        first_run = not tr.is_reader_loaded()
        hint = (
            "Loading OCR model (one-time ~85 MB download)..."
            if first_run
            else "Reading text locally..."
        )
        with st.spinner(hint):
            result = tr.extract_text(image_bytes)
        st.session_state[OCR_RESULT] = result
        st.session_state[OCR_RESULT_SRC] = image_bytes
        st.rerun()

    _render_ocr_results(image_bytes)


def _render_ocr_results(current_image_bytes: bytes) -> None:
    result: tr.TextRecognitionResult | None = st.session_state.get(OCR_RESULT)
    if result is None or st.session_state.get(OCR_RESULT_SRC) != current_image_bytes:
        return

    if result.error:
        components.status_banner("error", result.error)
        return

    if result.is_empty:
        # A valid, successful run that found no text - never an error.
        components.status_banner(
            "info",
            "No text found. The image is valid; it may contain handwriting, "
            "very small print, heavy blur, or no text at all. OCR supports "
            "English only for now.",
        )
        return

    components.status_banner(
        "success",
        f"Text extracted ({len(result.regions)} region(s), "
        f"{result.processing_ms} ms on CPU).",
    )

    st.subheader("Extracted text")
    st.markdown(result.text)
    st.code(result.text, language=None)  # built-in copy button (copy control)
    st.download_button(
        "Download text", data=result.text, file_name="extracted_text.txt"
    )

    avg_conf = sum(r.confidence for r in result.regions) / len(result.regions)
    st.caption(
        f"Average model confidence: {avg_conf * 100:.0f}%. Confidence is a "
        "model score, not a correctness guarantee - please verify important "
        "text. Handwriting and very low quality images may cause errors."
    )

    with st.expander("Region details"):
        for i, region in enumerate(result.regions, 1):
            st.markdown(f"- Region {i}: **{region.text}** ({region.confidence * 100:.0f}%)")

    # Read-aloud reuses the shared voice assistant (Phase 3 controls)
    tts_ok = vas.is_available()
    if not tts_ok:
        components.status_banner("warning", vas.setup_hint())
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Speak text", type="primary", disabled=not tts_ok):
        if vas.assistant.speak(result.text):
            st.session_state["ocr_speech_status"] = "Speaking extracted text. Use Stop to interrupt."
        else:
            st.session_state["ocr_speech_status"] = "Speech could not start."
    b2.button("Stop speech", disabled=not tts_ok, key="ocr_stop_speech")
    if st.session_state.get("ocr_stop_speech"):
        vas.assistant.stop()
        st.session_state["ocr_speech_status"] = "Speech stopped."
    status = st.session_state.get("ocr_speech_status", "")
    if status:
        st.caption(status)
    elif tts_ok:
        st.caption("Not speaking. Press Speak text to start.")


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
            "- **OCR model:** EasyOCR English, loads on first use "
            "(one-time ~85 MB download, then offline)\n"
            "- **Network use:** one-time model downloads only\n"
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

    # Resource cleanup on page change: leaving Scene Assistant stops the
    # camera (easy reversal + guaranteed release, spec Phase 5).
    if st.session_state.get(LAST_PAGE) not in (None, page):
        _stop_camera()
    st.session_state[LAST_PAGE] = page

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
