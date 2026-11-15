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
from modules.image_validation import validate_upload
from modules.navigation import route_service as nav
from modules.position_analysis import describe_position
from ui import components, map_view, styles

st.set_page_config(
    page_title="AccessAble AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)
styles.inject_global_css()

APP_VERSION = "1.1.0"

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
SESSION_UPLOAD_EPOCH = "scene_upload_epoch"
SESSION_UPLOAD_FP = "scene_upload_fingerprint"

OCR_SOURCE = "ocr_image_bytes"
OCR_NAME = "ocr_image_name"
OCR_CLEAR = "ocr_confirm_clear"
OCR_RESULT = "ocr_result"     # tr.TextRecognitionResult
OCR_RESULT_SRC = "ocr_source"
OCR_UPLOAD_EPOCH = "ocr_upload_epoch"
OCR_UPLOAD_FP = "ocr_upload_fingerprint"

CAMERA = "camera_instance"        # cam_mod.CameraCapture
CAMERA_AUTO = "camera_auto_preview"
CAMERA_SNAPSHOT = "camera_snapshot_bytes"
CAMERA_RESULT = "camera_result"   # od.DetectionResult for the snapshot
LAST_PAGE = "last_active_page"


def _bump_epoch(key: str) -> None:
    """Reset a file uploader by changing its widget key (drops the file)."""
    st.session_state[key] = st.session_state.get(key, 0) + 1


def _uploader_key(base: str, epoch_key: str) -> str:
    return f"{base}_{st.session_state.get(epoch_key, 0)}"


def _navigate(page: str) -> None:
    """Navigation callback (on_click runs before widgets are instantiated)."""
    st.session_state["page"] = page


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="aa-appname">AccessAble AI</div>'
            '<div class="aa-apptag">Offline visual assistant prototype</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "Navigation",
            PAGES,
            key="page",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            '<div class="aa-apptag" style="display:flex;gap:8px;'
            'align-items:flex-start;line-height:1.45;">'
            f'{components.svg_icon("shield", colour="#5EEAD4", size=16)}'
            "<span>All processing stays on this machine.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="aa-apptag" style="margin-top:14px;">'
            f"Prototype v{APP_VERSION} · offline · CPU</div>",
            unsafe_allow_html=True,
        )
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
            "eye",
            "Scene Assistant",
            "Upload an image to detect common objects with a small pretrained "
            "YOLO model running on your CPU, with confidence values and "
            "relative-position descriptions.",
        )
        components.feature_card(
            "text",
            "Read Text",
            "Extract printed text from an image with local OCR and have it "
            "spoken aloud. Loads only when you request it.",
        )
    with c2:
        components.feature_card(
            "speaker",
            "Voice feedback",
            "Concise spoken summaries using offline text-to-speech, with "
            "speak, stop, replay, and rate controls. A visual text version "
            "is always provided.",
        )
        components.feature_card(
            "map",
            "Indoor Navigation Demo",
            "A clearly fictional sample map used to demonstrate BFS and A* "
            "route planning. Simulation only - never live guidance.",
        )

    st.subheader("System readiness")
    model_ready = od.is_model_loaded()
    speech_ready = vas.is_available()
    p1, p2, p3 = st.columns(3)
    with p1:
        components.status_pill(
            "Detection model",
            "cached" if model_ready else "loads on first use",
            model_ready,
        )
    with p2:
        components.status_pill(
            "Speech engine",
            "ready" if speech_ready else "unavailable",
            speech_ready,
        )
    with p3:
        components.status_pill("Privacy", "fully local", True)

    st.button(
        "Open Scene Assistant",
        type="primary",
        on_click=_navigate,
        args=("Scene Assistant",),
    )


# ---------------------------------------------------------------------------
# Scene Assistant (Phase 1: upload workflow; Phase 2 adds real detection)
# ---------------------------------------------------------------------------
def _load_sample(path: str) -> None:
    with open(path, "rb") as fh:
        st.session_state[SESSION_IMAGE] = fh.read()
    st.session_state[SESSION_NAME] = path.rsplit("/", 1)[-1]
    st.session_state.pop(SESSION_CLEAR, None)
    st.session_state.pop(SESSION_UPLOAD_FP, None)
    _bump_epoch(SESSION_UPLOAD_EPOCH)  # drop any file held by the uploader
    st.rerun()


def _clear_image() -> None:
    vas.assistant.stop()  # easy reversal: no speech continues over cleared content
    for key in (SESSION_IMAGE, SESSION_NAME, SESSION_CLEAR, SESSION_RESULT,
                SESSION_RESULT_SRC, SESSION_RESULT_CONF, SESSION_UPLOAD_FP):
        st.session_state.pop(key, None)
    _bump_epoch(SESSION_UPLOAD_EPOCH)  # reset the uploader widget itself
    st.rerun()


def render_scene_assistant() -> None:
    components.page_header(
        "Scene Assistant",
        "Detect common objects in an uploaded image or a live camera snapshot, "
        "then hear a concise spoken summary. Everything runs locally.",
    )
    components.privacy_notice()

    mode = st.radio(
        "Choose how to provide an image",
        ["Image upload", "Live camera"],
        horizontal=True,
        key="scene_mode",
    )
    _render_voice_settings()

    if mode == "Live camera":
        # The camera must only run while its mode is on screen.
        render_camera_section()
        return

    if st.session_state.get(CAMERA) is not None:
        _stop_camera()  # leaving camera mode releases the device

    _render_upload_mode()


def _render_upload_mode() -> None:
    uploaded = st.file_uploader(
        "Upload an image (JPG or PNG, up to 10 MB)",
        type=["jpg", "jpeg", "png"],
        key=_uploader_key("scene_uploader", SESSION_UPLOAD_EPOCH),
    )

    if uploaded is not None:
        fingerprint = (uploaded.name, uploaded.size)
        # Only adopt a newly presented file - never override a sample or a
        # cleared selection with a file the uploader is still holding.
        if st.session_state.get(SESSION_UPLOAD_FP) != fingerprint:
            error = validate_upload(uploaded.name, uploaded.size)
            data = None
            if error is None:
                data = uploaded.getvalue()
                error = validate_upload(uploaded.name, uploaded.size, data)
            if error:
                components.status_banner("error", error)
            else:
                st.session_state[SESSION_IMAGE] = data
                st.session_state[SESSION_NAME] = uploaded.name
                st.session_state[SESSION_UPLOAD_FP] = fingerprint
                st.session_state.pop(SESSION_CLEAR, None)

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
                width="stretch",
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
        key="conf_threshold",
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


def _render_voice_settings() -> bool:
    """Single shared speech-rate control, rendered once per page."""
    tts_ok = vas.is_available()
    with st.expander("Voice settings", expanded=False):
        if not tts_ok:
            components.status_banner("warning", vas.setup_hint())
        rate = st.slider(
            "Speech rate (words per minute)",
            min_value=100,
            max_value=220,
            value=vas.assistant.rate,
            step=10,
            key="speech_rate",
            format="%d words/min",
            disabled=not tts_ok,
            help="Applies to the next utterance; shared across all pages.",
        )
        vas.assistant.set_rate(rate)
    return tts_ok


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
    if result.annotated_png and result.detections:
        show_boxes = st.checkbox(
            "Show bounding boxes on preview", value=True,
            help="Uncheck to view the original image instead of the overlay.",
        )
        if show_boxes:
            st.image(
                Image.open(io.BytesIO(result.annotated_png)),
                caption="Detected objects in this image",
                width="stretch",
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
                "Bounding box (x1, y1, x2, y2)": str(d.bbox),
            }
            for d in result.detections
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
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
    _render_speech_controls(result, prefix="upload")


def _render_speech_controls(
    result: od.DetectionResult, *, prefix: str, heading: str = "Spoken summary"
) -> None:
    """Concise spoken summary with speak / stop / replay controls.

    Speech is never started automatically (internal locus of control);
    the text equivalent is always on screen (audio never the only channel).
    Widget keys are prefixed so two panels can coexist on one page.
    """
    if not result.detections:
        return

    summary = am.summarize(result.detections, result.image_size[0], result.image_size[1])
    tts_ok = vas.is_available()
    status_key = f"speech_status_{prefix}"

    st.subheader(heading)
    with st.container(border=True):
        st.markdown(f"**Text version:** {summary.summary}")
        if not tts_ok:
            components.status_banner("warning", vas.setup_hint())

        b1, b2, b3 = st.columns(3)
        speak_pressed = b1.button(
            "Speak summary", type="primary", disabled=not tts_ok,
            key=f"{prefix}_speak",
        )
        stop_pressed = b2.button(
            "Stop speech", disabled=not tts_ok, key=f"{prefix}_stop",
        )
        replay_pressed = b3.button(
            "Replay speech",
            disabled=not tts_ok or not vas.assistant.has_last_text(),
            key=f"{prefix}_replay",
        )

        if speak_pressed:
            if vas.assistant.speak(summary.summary):
                st.session_state[status_key] = "Speaking. Use Stop to interrupt."
            else:
                st.session_state[status_key] = "Speech could not start."
            st.rerun()
        if stop_pressed:
            vas.assistant.stop()
            st.session_state[status_key] = "Speech stopped."
            st.rerun()
        if replay_pressed:
            if vas.assistant.replay():
                st.session_state[status_key] = "Replaying last summary."
            else:
                st.session_state[status_key] = "Speech could not start."
            st.rerun()

        status = st.session_state.get(status_key, "")
        if status:
            if vas.assistant.is_speaking():
                st.info(status, icon="🗣️")
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
    st.caption(
        "Camera frames are processed locally in memory and are never saved, "
        "uploaded, or transmitted. The camera stays OFF until you press Start."
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
        "stops automatically when you leave this page or mode.",
    )
    if cam_inst.get_error():
        components.status_banner("error", cam_inst.get_error())
        _stop_camera()
        st.rerun()

    auto = st.checkbox(
        "Live preview (auto-refresh)",
        key=CAMERA_AUTO,
        help="Refreshes the preview every interval. Inference still runs "
        "only when you press Capture & analyze.",
    )
    if auto:
        st.slider(
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
            width="stretch",
        )

    threshold = st.session_state.get("conf_threshold", od.DEFAULT_CONFIDENCE)
    st.caption(
        f"Detection threshold for snapshots: **{threshold:.2f}** "
        "(change it in the Image upload mode)."
    )

    b0, b1, b2, _ = st.columns([1.2, 1.5, 1.4, 1.6])
    refresh_pressed = False
    if not auto:
        refresh_pressed = b0.button("Refresh preview", key="camera_refresh")
    capture_pressed = b1.button(
        "Capture & analyze", type="primary", key="camera_capture",
    )
    stop_pressed = b2.button("Stop camera", key="camera_stop")

    if stop_pressed:
        _stop_camera()
        st.rerun()

    if capture_pressed:
        if frame is None:
            components.status_banner("warning", "No frame available yet.")
        else:
            snapshot = _encode_frame_png(frame)
            with st.spinner("Analyzing snapshot..."):
                result = od.analyze_image(
                    snapshot, confidence_threshold=threshold,
                )
            st.session_state[CAMERA_SNAPSHOT] = snapshot
            st.session_state[CAMERA_RESULT] = result
            st.rerun()

    _render_camera_results()

    # Throttled preview loop: capture rate (15 fps in the grabber thread)
    # stays decoupled from the UI refresh and from inference.
    if (auto or refresh_pressed) and cam_inst.is_active():
        if auto:
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
            width="stretch",
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
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        components.status_banner(
            "info",
            "No objects detected in this snapshot. The model recognizes only "
            "its training classes (80 COCO classes).",
        )
    _render_speech_controls(result, prefix="snapshot", heading="Spoken snapshot summary")


# ---------------------------------------------------------------------------
# Read Text (Phase 4: real local OCR)
# ---------------------------------------------------------------------------
def _load_ocr_sample(path: str) -> None:
    with open(path, "rb") as fh:
        st.session_state[OCR_SOURCE] = fh.read()
    st.session_state[OCR_NAME] = path.rsplit("/", 1)[-1]
    st.session_state.pop(OCR_CLEAR, None)
    st.session_state.pop(OCR_UPLOAD_FP, None)
    _bump_epoch(OCR_UPLOAD_EPOCH)  # drop any file held by the uploader
    st.rerun()


def _clear_ocr_image() -> None:
    vas.assistant.stop()  # easy reversal: no speech continues over cleared content
    for key in (OCR_SOURCE, OCR_NAME, OCR_CLEAR, OCR_RESULT, OCR_RESULT_SRC,
                OCR_UPLOAD_FP):
        st.session_state.pop(key, None)
    _bump_epoch(OCR_UPLOAD_EPOCH)  # reset the uploader widget itself
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
        key=_uploader_key("ocr_uploader", OCR_UPLOAD_EPOCH),
    )
    if uploaded is not None:
        fingerprint = (uploaded.name, uploaded.size)
        if st.session_state.get(OCR_UPLOAD_FP) != fingerprint:
            error = validate_upload(uploaded.name, uploaded.size)
            data = None
            if error is None:
                data = uploaded.getvalue()
                error = validate_upload(uploaded.name, uploaded.size, data)
            if error:
                components.status_banner("error", error)
            else:
                st.session_state[OCR_SOURCE] = data
                st.session_state[OCR_NAME] = uploaded.name
                st.session_state[OCR_UPLOAD_FP] = fingerprint
                st.session_state.pop(OCR_CLEAR, None)

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
            st.image(Image.open(io.BytesIO(image_bytes)), width="stretch")
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
    st.code(result.text, language=None)  # faithful, selectable, copyable
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
    tts_ok = _render_voice_settings()
    b1, b2, _ = st.columns([1, 1, 2])
    speak_pressed = b1.button(
        "Speak text", type="primary", disabled=not tts_ok, key="ocr_speak",
    )
    stop_pressed = b2.button(
        "Stop speech", disabled=not tts_ok, key="ocr_stop",
    )
    if speak_pressed:
        if vas.assistant.speak(result.text):
            st.session_state["ocr_speech_status"] = (
                "Speaking extracted text. Use Stop to interrupt."
            )
        else:
            st.session_state["ocr_speech_status"] = "Speech could not start."
        st.rerun()
    if stop_pressed:
        vas.assistant.stop()
        st.session_state["ocr_speech_status"] = "Speech stopped."
        st.rerun()
    status = st.session_state.get("ocr_speech_status", "")
    if status:
        if vas.assistant.is_speaking():
            st.info(status, icon="🗣️")
        else:
            st.caption(status)
    elif tts_ok and not vas.assistant.is_speaking():
        st.caption("Not speaking. Press Speak text to start.")


# ---------------------------------------------------------------------------
# Indoor Navigation Demo (Phase 6) - simulation on a fictional sample map
# ---------------------------------------------------------------------------
def render_navigation() -> None:
    components.page_header(
        "Indoor Navigation Demo",
        "Explore BFS and A* route planning on a fictional sample building map.",
    )
    components.safety_notice()  # mandatory on the navigation page
    components.status_banner(
        "warning",
        "Simulation only. This page uses a FIXED FICTIONAL MAP - it does "
        "not know your live location and cannot detect stairs, doors, "
        "temporary obstacles, or step-free routes. It is an algorithm "
        "demonstration, not live guidance.",
    )

    graph = nav.get_graph()
    names = graph.ids()

    c1, c2 = st.columns(2)
    with c1:
        start = st.selectbox(
            "Start location", names,
            index=names.index("entrance"),
            format_func=graph.display_name,
        )
    with c2:
        # default destination: science_lab (far corner)
        dest = st.selectbox(
            "Destination", names,
            index=names.index("science_lab"),
            format_func=graph.display_name,
        )

    algorithm = st.radio(
        "Algorithm", ["BFS (fewest hops)", "A* (shortest distance)", "Compare both"],
        horizontal=True,
        help="BFS minimizes the number of hops; A* minimizes walking "
        "distance using an admissible straight-line heuristic.",
    )

    if st.button("Find route", type="primary"):
        st.session_state["nav_request"] = (start, dest, algorithm)
        st.rerun()

    request = st.session_state.get("nav_request")
    if not request:
        st.caption(
            "Pick a start and destination, then press Find route. The map "
            "and its format are documented in assets/sample_maps/."
        )
        return

    if request != (start, dest, algorithm):
        components.status_banner(
            "warning",
            "Selections changed since this route was calculated. Press "
            "Find route to update the results below.",
        )

    start, dest, algorithm = request
    if algorithm == "Compare both":
        bfs_result, astar_result = nav.compare_routes(graph, start, dest)
        _render_route_result(bfs_result, graph, with_map=False)
        st.divider()
        _render_route_result(astar_result, graph, with_map=False)
        _render_comparison(bfs_result, astar_result)
        _render_comparison_maps(bfs_result, astar_result, graph)
    else:
        algo_key = "BFS" if algorithm.startswith("BFS") else "A*"
        _render_route_result(nav.run_route(graph, start, dest, algo_key), graph)


def _render_route_result(
    result: nav.RouteResult, graph, *, with_map: bool = True
) -> None:
    title = result.algorithm if result.algorithm else "Route"
    st.subheader(f"{title} result")

    if result.error:
        components.status_banner("error", result.error)
        return

    if result.unreachable:
        components.status_banner(
            "info",
            f"No route exists between {graph.display_name(result.start)} "
            f"and {graph.display_name(result.goal)} on this map "
            f"({result.explored_count} nodes explored before concluding this). "
            "This is a correct algorithmic result for the disconnected part "
            "of the sample map.",
        )
        return

    components.status_banner(
        "success",
        f"Route found: {result.hops} hop(s), {result.cost:.0f} m, "
        f"{result.explored_count} nodes explored, {result.time_ms} ms.",
    )

    if with_map and result.path:
        with st.container(border=True):
            map_view.render_route_map(
                graph,
                result.path,
                caption=(
                    "Fictional sample map - schematic layout of the demo graph, "
                    "not a real building. Start (green), destination (amber), "
                    "route (teal). The written route below is the primary "
                    "description."
                ),
            )

    # Route as a readable flow, then a step table with cumulative distance.
    st.markdown(" → ".join(f"**{name}**" for name in result.path_display))

    with st.expander("Step-by-step route"):
        cumulative = 0.0
        for i, (node_id, name) in enumerate(zip(result.path, result.path_display)):
            if i > 0:
                for neighbour, weight in graph.neighbours(result.path[i - 1]):
                    if neighbour == node_id:
                        cumulative += weight
                        break
            st.markdown(
                f"**{i + 1}. {name}**"
                + (f" — {cumulative:.0f} m from start" if i > 0 else " — starting point")
            )

    with st.expander(f"Nodes explored ({result.explored_count})"):
        st.caption(" → ".join(result.explored_names))

    st.caption(
        "These numbers describe the fictional sample map only - not any "
        "real building, and never safe real-world guidance."
    )


def _render_comparison_maps(
    bfs_result: nav.RouteResult,
    astar_result: nav.RouteResult,
    graph,
) -> None:
    if not (bfs_result.found and astar_result.found):
        return
    st.subheader("Route maps")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**BFS - fewest hops**")
        map_view.render_route_map(graph, bfs_result.path)
    with c2:
        st.markdown("**A* - shortest distance**")
        map_view.render_route_map(graph, astar_result.path)
    st.caption(
        "Both maps come from the fictional sample graph. When the two routes "
        "differ, note how BFS prefers fewer stops while A* may take more hops "
        "to walk a shorter distance. Schematic layout - not a real building."
    )


def _render_comparison(bfs_result: nav.RouteResult, astar_result: nav.RouteResult) -> None:
    st.subheader("BFS vs A* comparison")
    if not (bfs_result.found and astar_result.found):
        return
    st.dataframe(
        [
            {
                "Algorithm": "BFS",
                "Hops": bfs_result.hops,
                "Distance (m)": round(bfs_result.cost, 1),
                "Nodes explored": bfs_result.explored_count,
                "Time (ms)": bfs_result.time_ms,
            },
            {
                "Algorithm": "A*",
                "Hops": astar_result.hops,
                "Distance (m)": round(astar_result.cost, 1),
                "Nodes explored": astar_result.explored_count,
                "Time (ms)": astar_result.time_ms,
            },
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "BFS minimizes hops and ignores distances; A* minimizes walking "
        "distance using the straight-line heuristic and typically explores "
        "fewer nodes. Both are correct for the sample map - they simply "
        "optimize different things."
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
        "Speech rate (words per minute)",
        min_value=100,
        max_value=220,
        value=vas.assistant.rate,
        step=10,
        key="speech_rate",
        format="%d words/min",
        help="Shared with the Voice settings control on the Scene Assistant "
        "and Read Text pages; applies to the next spoken utterance.",
    )
    vas.assistant.set_rate(st.session_state.get("speech_rate", vas.DEFAULT_RATE))
    current_conf = st.session_state.get("conf_threshold", od.DEFAULT_CONFIDENCE)
    st.caption(
        f"Detection confidence threshold: currently **{current_conf:.2f}** - "
        "change it on the Scene Assistant page (Image upload mode)."
    )

    st.subheader("Model / device status")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            "- **Device:** CPU only (GPU optional, not required)\n"
            "- **Detection model:** yolo11n nano (Ultralytics, AGPL-3.0), "
            "cached locally after first use\n"
            "- **Speech:** eSpeak NG offline text-to-speech\n"
            "- **Navigation:** hand-written BFS and A* on a fixed sample map"
        )
    with d2:
        st.markdown(
            "- **OCR model:** EasyOCR English, loads on first use "
            "(one-time ~85 MB download, then offline)\n"
            "- **Network use:** one-time model downloads only\n"
            "- **Telemetry:** disabled (Streamlit usage stats off)\n"
            "- **Data retention:** none\n"
            f"- **Version:** {APP_VERSION} (educational prototype)"
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
    # camera (easy reversal + guaranteed release, spec Phase 5), and any
    # playing speech stops too - nothing continues behind the user's back.
    if st.session_state.get(LAST_PAGE) not in (None, page):
        _stop_camera()
        vas.assistant.stop()
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
