"""AccessAble AI - application entry point.

Thin UI shell. Business logic lives in modules/; shared visual pieces in
ui/. Routing is a simple sidebar radio (predictable, keyboard-friendly).

Pages:
  Dashboard            - quick start, feature cards, readiness, notices
  Scene Assistant      - upload / camera detection, text and depth checks
  Read Text            - local OCR with read-aloud
  Indoor Navigation    - BFS / A* demo on a fictional map
  Settings & Help      - preferences, privacy, limitations
"""

from __future__ import annotations

import io
import time

import streamlit as st
from PIL import Image

from modules import announcement_manager as am
from modules import camera as cam_mod
from modules import depth_estimation as depth_mod
from modules import object_detection as od
from modules import open_vocabulary as ov
from modules import rubiks_detector as rubiks
from modules import runtime
from modules import scene_tracker as tracker
from modules import text_recognition as tr
from modules import voice_assistant as vas
from modules.image_validation import validate_upload
from modules.navigation import instructions as nav_steps
from modules.navigation import profiles as nav_profiles
from modules.navigation import route_service as nav
from modules.navigation import scenarios as nav_scenarios
from modules.navigation.graph_data import floor_label as nav_floor_label
from modules.position_analysis import describe_position
from ui import browser_speech, components, map_view, styles

st.set_page_config(
    page_title="AccessAble AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)
styles.inject_global_css()

APP_VERSION = "1.4.0"

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
SESSION_RESULT_OPTS = "detection_opts"   # (model kind, vocabulary) of result
DEPTH_ERROR = "depth_error"
CUBE_NOTE = "rubiks_note"  # (kind, message) from the Rubik's cube check
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
CAMERA_BRIEF = "camera_auto_brief"
CAMERA_BRIEF_INTERVAL = "brief_interval"
CAMERA_BRIEF_SPEAK = "brief_speak"
BRIEF_LOG = "brief_log"           # list[str], newest first
BRIEF_MEMORY = "brief_memory"     # (camera instance, tracker.SceneMemory)
LAST_PAGE = "last_active_page"

NAV_START = "nav_start"
NAV_DEST = "nav_dest"
NAV_PROFILE = "nav_profile"
NAV_SPEED = "nav_speed"
NAV_SCENARIO = "nav_scenario"
NAV_CLOSURES = "nav_closures"
NAV_CLICK_MODE = "nav_click_mode"
NAV_MAP_EPOCH = "nav_map_epoch"
NAV_PENDING = "nav_pending"
NAV_STEP = "nav_step"
NAV_STEP_SIG = "nav_step_sig"
NAV_STEP_COUNT = "nav_step_count"
NAV_PLAYING = "nav_playing"
NAV_LAST_ADVANCE = "nav_last_advance"
NAV_SPEAK_STEPS = "nav_speak_steps"
NAV_LAST_SPOKEN = "nav_last_spoken"
NAV_SHOW_EXPLORED = "nav_show_explored"
NAV_SHOW_LABELS = "nav_show_labels"
NAV_FLASH = "nav_flash"


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
            "<span>Images are processed in memory and never stored.</span></div>",
            unsafe_allow_html=True,
        )
        mode_label = "hosted" if runtime.is_hosted() else "offline"
        st.markdown(
            f'<div class="aa-apptag" style="margin-top:14px;">'
            f"Prototype v{APP_VERSION} · {mode_label} · CPU</div>",
            unsafe_allow_html=True,
        )
    return page


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    components.page_header(
        "Welcome to AccessAble AI",
        "Understand a scene, read visible text, and explore indoor routes - no cloud AI APIs.",
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
            "Upload an image or use a live camera to detect common objects, "
            "search for custom items (door, stairs, ramp) with open-vocabulary "
            "detection, and optionally estimate relative proximity.",
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
            (
                "Concise spoken summaries using your browser's voices, with "
                "speak, stop, replay, and rate controls. A visual text version "
                "is always provided."
                if runtime.is_hosted()
                else "Concise spoken summaries using offline text-to-speech, "
                "with speak, stop, replay, and rate controls. A visual text "
                "version is always provided."
            ),
        )
        components.feature_card(
            "map",
            "Indoor Navigation Demo",
            "Explore a fictional two-floor map: click to set start or "
            "destination, close segments, compare Fastest, Step-free, and "
            "Quiet profiles, and follow a spoken walk-through. Simulation "
            "only - never live guidance.",
        )

    st.subheader("System readiness")
    model_ready = od.is_model_loaded()
    speech_ready = True if runtime.is_hosted() else vas.is_available()
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
            (
                "browser voices"
                if runtime.is_hosted()
                else ("ready" if speech_ready else "unavailable")
            ),
            speech_ready,
        )
    with p3:
        components.status_pill(
            "Privacy",
            "no storage" if runtime.is_hosted() else "fully local",
            True,
        )

    st.button(
        "Open Scene Assistant",
        type="primary",
        on_click=_navigate,
        args=("Scene Assistant",),
    )


# ---------------------------------------------------------------------------
# Scene Assistant
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
        (
            "Detect common objects in an uploaded image or a browser camera "
            "snapshot, then hear a concise spoken summary. Processing stays "
            "in memory."
            if runtime.is_hosted()
            else "Detect common objects in an uploaded image or a live camera "
            "snapshot, then hear a concise spoken summary. Processing stays "
            "in memory."
        ),
    )
    components.privacy_notice()

    modes = (
        ["Image upload", "Camera snapshot"]
        if runtime.is_hosted()
        else ["Image upload", "Live camera"]
    )
    mode = st.radio(
        "Choose how to provide an image",
        modes,
        horizontal=True,
        key="scene_mode",
    )
    _render_voice_settings()

    if mode == "Camera snapshot":
        # A hosted server cannot reach the visitor's webcam; the browser
        # takes one photo and sends it to the server for analysis.
        _render_browser_camera()
        return

    if mode == "Live camera":
        # The camera must only run while its mode is on screen.
        render_camera_section()
        return

    if st.session_state.get(CAMERA) is not None:
        _stop_camera()  # leaving camera mode releases the device

    _render_upload_mode()


def _render_browser_camera() -> None:
    """Browser camera snapshot mode for hosted deployments."""
    st.caption(
        "Uses your browser's camera over HTTPS. The photo is sent to the app "
        "server for processing and is not saved. Browser camera access "
        "requires camera permission."
    )
    shot = st.camera_input("Take a photo", key="browser_camera_photo")
    if shot is not None:
        data = shot.getvalue()
        error = validate_upload(shot.name or "camera.jpg", shot.size or len(data), data)
        if error:
            components.status_banner("error", error)
        else:
            threshold = st.session_state.get(
                "conf_threshold", od.DEFAULT_CONFIDENCE
            )
            st.caption(
                f"Detection threshold: **{threshold:.2f}** - change it in "
                "Image upload mode."
            )
            if st.button(
                "Analyze this photo", type="primary", key="browser_camera_analyze"
            ):
                with st.spinner("Analyzing on CPU..."):
                    result = od.analyze_image(
                        data, confidence_threshold=threshold
                    )
                st.session_state[CAMERA_SNAPSHOT] = data
                st.session_state[CAMERA_RESULT] = result
                st.rerun()

    _render_camera_results()


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

    # Stale-result guard: if the image or detection settings changed since
    # the last analysis, old results must not be presented as current.
    saved_opts = st.session_state.get(SESSION_RESULT_OPTS)
    model_choice = st.session_state.get(
        "detection_model", "Standard (80 everyday objects)"
    )
    if str(model_choice).startswith("Open"):
        opts_now = (
            "open",
            tuple(
                ov.parse_vocabulary(
                    st.session_state.get("world_vocabulary", ov.DEFAULT_VOCABULARY)
                )
            ),
        )
    else:
        opts_now = ("standard", ())
    opts_now = opts_now + (bool(st.session_state.get("rubiks_check", False)),)
    stale = st.session_state.get(SESSION_RESULT) is not None and (
        st.session_state.get(SESSION_RESULT_SRC) != image_bytes
        or saved_opts != opts_now
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
            "The image or detection settings changed since the last "
            "analysis. Press Analyze image again to refresh the results.",
        )

    # -- Detection settings (standard vs open vocabulary + depth) -------------
    st.divider()
    st.markdown("#### Detection settings")
    model_label = st.radio(
        "Detection model",
        [
            "Standard (80 everyday objects)",
            "Open vocabulary (describe what to find)",
        ],
        horizontal=True,
        key="detection_model",
        help="The standard model is fast and reliable for its 80 classes. "
        "Open vocabulary can look for doors, stairs, ramps, and more - with "
        "less reliable results.",
    )
    use_open_vocab = model_label.startswith("Open")

    vocabulary_text = ""
    if use_open_vocab:
        vocabulary_text = st.text_area(
            "Objects to look for (comma-separated, up to 30)",
            value=st.session_state.get("world_vocabulary", ov.DEFAULT_VOCABULARY),
            key="world_vocabulary",
            height=100,
            help="YOLO-World downloads once (~380 MB: model + CLIP text "
            "encoder) then runs offline. Scores tend to be lower than the "
            "standard model.",
        )
        st.caption(
            "Tip: open-vocabulary scores run lower - try a threshold around "
            "0.15-0.25. Unusual phrases may produce false positives."
        )

    use_depth = st.checkbox(
        "Estimate relative proximity (depth)",
        key="use_depth",
        help="Adds near/far bands within this image only (one-time ~100 MB "
        "download). Relative, not metres - never a safety signal.",
    )

    use_rubiks = st.checkbox(
        "Rubik's cube face check (3x3 colour grid)",
        key="rubiks_check",
        help="Heuristic pattern check for a scrambled cube face. Not a "
        "trained model and not 100% accurate: solved single-colour faces, "
        "heavy glare, and very small cubes can be missed. Image upload only.",
    )

    if use_open_vocab:
        conf_threshold = st.slider(
            "Open-vocabulary confidence threshold",
            min_value=0.02,
            max_value=0.60,
            value=0.10,
            step=0.01,
            format="%.2f",
            key="world_conf",
            help=(
                "Open-vocabulary scores run lower than the standard model. "
                "Confidence is a model score, not a probability of safety."
            ),
        )
    else:
        conf_threshold = st.slider(
            "Detection confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.40,
            step=0.05,
            format="%.2f",
            key="conf_threshold",
            help=(
                "Minimum model score for a detection to be shown. Confidence "
                "is a model score, not a probability of safety."
            ),
        )

    current_opts = (
        "open" if use_open_vocab else "standard",
        tuple(ov.parse_vocabulary(vocabulary_text)) if use_open_vocab else (),
        use_rubiks,
    )

    if st.button("Analyze image", type="primary"):
        classes = list(current_opts[1])
        if use_open_vocab and not classes:
            components.status_banner(
                "error",
                "Add at least one object name to look for (for example: door, "
                "stairs, ramp).",
            )
        else:
            if use_open_vocab:
                first_run = not ov.is_model_loaded()
                hint = (
                    "Loading open-vocabulary model (one-time ~380 MB "
                    "download)..." if first_run
                    else "Searching for your objects..."
                )
            else:
                first_run = not od.is_model_loaded()
                hint = (
                    "Loading model (one-time ~5 MB download)..." if first_run
                    else "Detecting objects..."
                )
            with st.spinner(hint):
                if use_open_vocab:
                    result = ov.analyze_image(
                        image_bytes, classes, confidence_threshold=conf_threshold
                    )
                else:
                    result = od.analyze_image(
                        image_bytes, confidence_threshold=conf_threshold
                    )
                st.session_state.pop(DEPTH_ERROR, None)
                if use_depth and result.detections and not result.error:
                    depth_hint = (
                        "Loading depth model (one-time ~100 MB download)..."
                        if not depth_mod.is_loaded()
                        else "Estimating relative proximity..."
                    )
                    with st.spinner(depth_hint):
                        depth_error = depth_mod.annotate_result(result, image_bytes)
                    if depth_error:
                        st.session_state[DEPTH_ERROR] = depth_error
            st.session_state.pop(CUBE_NOTE, None)
            if use_rubiks:
                cube, cube_error = rubiks.detect_rubiks_cube(image_bytes)
                if cube_error:
                    st.session_state[CUBE_NOTE] = ("warning", cube_error)
                elif cube is not None:
                    result.detections = sorted(
                        [*result.detections, cube],
                        key=lambda d: d.confidence,
                        reverse=True,
                    )
                    try:
                        result.annotated_png = od.annotate_image(
                            image_bytes, result.detections
                        )
                    except Exception:
                        pass  # detections remain valid without the overlay
                    st.session_state[CUBE_NOTE] = (
                        "success",
                        (
                            "Rubik's cube check matched a 3x3 colour grid "
                            f"(heuristic pattern score {cube.confidence:.0%}; "
                            "not a trained model, not 100% accurate)."
                        ),
                    )
                else:
                    st.session_state[CUBE_NOTE] = (
                        "info",
                        (
                            "Rubik's cube check: no 3x3 colour grid found. A "
                            "solved single-colour face, heavy glare, or a very "
                            "small cube may be missed."
                        ),
                    )
            st.session_state[SESSION_RESULT] = result
            st.session_state[SESSION_RESULT_SRC] = image_bytes
            st.session_state[SESSION_RESULT_CONF] = conf_threshold
            st.session_state[SESSION_RESULT_OPTS] = current_opts
            st.rerun()

    _render_detection_results(image_bytes, current_opts)


def _render_voice_settings() -> bool:
    """Single shared speech-rate control, rendered once per page."""
    tts_ok = True if runtime.is_hosted() else vas.is_available()
    with st.expander("Voice settings", expanded=False):
        if runtime.is_hosted():
            st.caption(
                "Speech is produced by your browser's built-in voices; no "
                "server audio setup is needed. Availability varies by browser "
                "and operating system."
            )
        elif not tts_ok:
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


def _render_detection_results(
    current_image_bytes: bytes, current_opts: tuple
) -> None:
    """Render the stored analysis result with all required states."""
    result: od.DetectionResult | None = st.session_state.get(SESSION_RESULT)
    if result is None or st.session_state.get(SESSION_RESULT_SRC) != current_image_bytes:
        return  # no result yet, or results belong to a previous image
    if st.session_state.get(SESSION_RESULT_OPTS) != current_opts:
        return  # settings changed; old results are stale and hidden

    conf_used = st.session_state.get(SESSION_RESULT_CONF, od.DEFAULT_CONFIDENCE)

    if result.error:
        components.status_banner("error", result.error)
        if not result.detections:
            return

    depth_error = st.session_state.get(DEPTH_ERROR)
    if depth_error:
        components.status_banner("warning", depth_error)

    cube_note = st.session_state.get(CUBE_NOTE)
    if cube_note:
        components.status_banner(cube_note[0], cube_note[1])

    if result.detections:
        banner = (
            f"{len(result.detections)} object(s) found - "
            f"{result.model_name} inference {result.inference_ms} ms on CPU "
            f"(threshold {conf_used:.2f}"
        )
        if result.depth_ms:
            banner += f", depth {result.depth_ms} ms"
        banner += ")."
        components.status_banner("success", banner)
    else:
        components.status_banner(
            "info",
            f"No objects detected above the threshold {conf_used:.2f}. "
            "Try lowering the threshold or using a clearer image. The "
            "standard model recognizes only its 80 COCO classes; the "
            "open-vocabulary model depends on your prompt wording.",
        )

    # Annotated preview with an optional overlay to reduce clutter
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

    # Structured detection list with image-relative position
    if result.detections:
        has_proximity = any(d.proximity for d in result.detections)
        has_heuristic = any(d.source != "model" for d in result.detections)
        st.subheader("Detected objects")
        rows = []
        for d in result.detections:
            row = {
                "Object": d.label.replace("_", " "),
                "Confidence": f"{d.confidence * 100:.0f}%",
                "Position in image": describe_position(
                    d, result.image_size[0], result.image_size[1]
                ).replace("appears on the ", "").replace(" side of the image", ""),
            }
            if has_proximity:
                row["Proximity (relative)"] = d.proximity or "n/a"
            if has_heuristic:
                row["Source"] = (
                    "colour-grid check" if d.source == "color-grid" else "model"
                )
            row["Bounding box (x1, y1, x2, y2)"] = str(d.bbox)
            rows.append(row)
        st.dataframe(rows, width="stretch", hide_index=True)
        caption = (
            "Confidence is the model's score for this image region - not a "
            "probability of safety. Position describes the uploaded image "
            "only, never your physical surroundings."
        )
        if has_proximity:
            caption += (
                " Proximity bands come from monocular depth and are relative "
                "within this single image - never metres, and never a safety "
                "signal."
            )
        st.caption(caption)
    else:
        st.caption(
            "Tip: lower the threshold above and analyze again. The pretrained "
            "model only knows its training classes, so valid images can "
            "legitimately produce no detections."
        )

    # -- Spoken summary: text always visible, speech user-triggered
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
    if runtime.is_hosted():
        _render_browser_speech_controls(
            summary.summary, prefix=prefix, heading=heading
        )
        return
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


def _render_browser_speech_controls(
    text: str, *, prefix: str, heading: str
) -> None:
    """Speak / stop / replay controls using the visitor's browser voices."""
    st.subheader(heading)
    with st.container(border=True):
        st.markdown(f"**Text version:** {text}")
        rate = st.session_state.get("speech_rate", vas.DEFAULT_RATE)
        last_key = f"browser_speech_last_{prefix}"
        last_text = st.session_state.get(last_key, "")

        b1, b2, b3 = st.columns(3)
        speak_pressed = b1.button(
            "Speak summary", type="primary", key=f"{prefix}_speak"
        )
        stop_pressed = b2.button("Stop speech", key=f"{prefix}_stop")
        replay_pressed = b3.button(
            "Replay speech",
            disabled=not last_text,
            key=f"{prefix}_replay",
        )

        if speak_pressed:
            st.session_state[last_key] = text
            browser_speech.speak(text, rate)
            st.caption("Speaking in this browser. Press Stop to cancel.")
        elif stop_pressed:
            browser_speech.stop()
            st.caption("Speech stopped.")
        elif replay_pressed:
            browser_speech.speak(last_text, rate)
            st.caption("Replaying the last summary.")
        else:
            st.caption(
                "Speech uses your browser's voices; the text version above is "
                "always available. Voices vary by browser and operating system."
            )


# ---------------------------------------------------------------------------
# Live camera mode - deliberate activation, snapshot and briefing inference
# ---------------------------------------------------------------------------
def _stop_camera() -> None:
    """Stop the camera and drop its state. Always safe to call."""
    cam_inst = st.session_state.get(CAMERA)
    if cam_inst is not None:
        cam_inst.stop()
    st.session_state.pop(CAMERA, None)
    st.session_state.pop(CAMERA_SNAPSHOT, None)
    st.session_state.pop(CAMERA_RESULT, None)
    st.session_state.pop(BRIEF_MEMORY, None)


def _encode_frame_png(frame) -> bytes | None:
    import cv2

    ok, buf = cv2.imencode(".png", frame)
    return buf.tobytes() if ok else None


def _encode_frame_jpeg(frame, quality: int = 85) -> bytes | None:
    import cv2

    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    return buf.tobytes() if ok else None


def render_camera_section() -> None:
    st.caption(
        "Camera frames are processed in memory and are never saved or "
        "shared. The camera stays OFF until you press Start."
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
        "Live preview",
        value=True,
        key=CAMERA_AUTO,
        help="Continuously refreshes the preview. Inference runs only on "
        "Capture & analyze, or at the auto-briefing interval below.",
    )
    interval = 0.25
    if auto:
        interval = st.slider(
            "Preview refresh interval (seconds)", 0.1, 1.0, 0.25, 0.05,
            key="preview_interval",
            format="%.2f s",
            help="Lower is smoother. Only the preview image refreshes - the "
            "rest of the page does not rerun.",
        )

    @st.fragment(run_every=interval if auto else None)
    def _live_preview() -> None:
        # A fragment re-renders only this image, not the whole page, so the
        # preview stays smooth even at several refreshes per second.
        if not cam_inst.is_active():
            st.rerun(scope="app")  # camera stopped elsewhere; rebuild page
        frame = cam_inst.get_frame()
        if frame is None:
            st.caption("Waiting for the first frame...")
            return
        st.image(
            frame[:, :, ::-1],  # BGR -> RGB for display
            caption="Live preview (uncaptured frames are not analyzed)",
            width="stretch",
            output_format="JPEG",
        )

    _live_preview()

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

    if refresh_pressed:
        st.rerun()

    if capture_pressed:
        frame = cam_inst.get_frame()  # freshest frame at press time
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

    # -- Auto-briefing (announce only changes, never every frame) ------------
    st.divider()
    st.subheader("Auto-briefing")
    st.caption(
        "Runs detection on the live preview at a set cadence and announces "
        "only objects that appear or clear. Every announcement is also "
        "shown as text below (audio is never the only channel)."
    )
    brief_on = st.checkbox(
        "Auto-briefing (announce changes)",
        key=CAMERA_BRIEF,
        help="New objects are confirmed over consecutive updates before "
        "being announced, so one-frame flicker is ignored.",
    )
    brief_interval = st.slider(
        "Briefing interval (seconds)", 1.0, 5.0, 2.5, 0.5,
        key=CAMERA_BRIEF_INTERVAL,
        disabled=not brief_on,
        help="How often detection runs on the latest frame. Shorter is more "
        "responsive but uses more CPU.",
    )
    speak_brief = st.checkbox(
        "Speak announcements",
        value=True,
        key=CAMERA_BRIEF_SPEAK,
        disabled=not brief_on
        or not (runtime.is_hosted() or vas.is_available()),
    )
    clear_pressed = st.button(
        "Clear briefing log", key="brief_clear", disabled=not brief_on,
    )
    if clear_pressed:
        st.session_state[BRIEF_LOG] = []
        st.session_state.pop(BRIEF_MEMORY, None)
        st.rerun()

    @st.fragment(run_every=brief_interval if brief_on else None)
    def _auto_brief() -> None:
        if not cam_inst.is_active():
            return
        frame = cam_inst.get_frame()
        if frame is None:
            st.caption("Waiting for the first frame...")
            return
        encoded = _encode_frame_jpeg(frame)
        if encoded is None:
            return
        brief_result = od.analyze_image(
            encoded, confidence_threshold=threshold
        )
        if brief_result.error and not brief_result.detections:
            st.caption(f"Auto-briefing paused: {brief_result.error}")
            return

        state = st.session_state.get(BRIEF_MEMORY)
        if not (isinstance(state, tuple) and state[0] is cam_inst):
            state = (cam_inst, tracker.SceneMemory())
            st.session_state[BRIEF_MEMORY] = state
        memory = state[1]
        report = memory.update(
            brief_result.detections,
            brief_result.image_size[0],
            brief_result.image_size[1],
        )
        if report.has_changes:
            log = list(st.session_state.get(BRIEF_LOG, []))
            log.insert(0, report.summary)
            st.session_state[BRIEF_LOG] = log[:8]
            if speak_brief:
                if runtime.is_hosted():
                    browser_speech.speak(
                        report.summary,
                        st.session_state.get("speech_rate", vas.DEFAULT_RATE),
                    )
                elif vas.is_available():
                    vas.assistant.speak(report.summary)

        log = st.session_state.get(BRIEF_LOG, [])
        if log:
            st.markdown("**Recent announcements**")
            for line in log[:5]:
                st.caption(line)
        else:
            st.caption(
                "Listening for changes - confirmed new objects appear here."
            )
        st.caption(
            f"Currently {len(brief_result.detections)} object(s) visible "
            f"(threshold {threshold:.2f}); changes only are announced."
        )

    if brief_on:
        _auto_brief()


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
# Read Text (local OCR)
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
            else "Reading text..."
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

    if runtime.is_hosted():
        # Browser speech output; the text remains selectable and copyable.
        tts_ok = _render_voice_settings()
        rate = st.session_state.get("speech_rate", vas.DEFAULT_RATE)
        b1, b2, _ = st.columns([1, 1, 2])
        speak_pressed = b1.button(
            "Speak text", type="primary", disabled=not tts_ok, key="ocr_speak",
        )
        stop_pressed = b2.button(
            "Stop speech", disabled=not tts_ok, key="ocr_stop",
        )
        if speak_pressed:
            browser_speech.speak(result.text, rate)
            st.caption("Speaking in this browser. Press Stop to cancel.")
        elif stop_pressed:
            browser_speech.stop()
            st.caption("Speech stopped.")
        else:
            st.caption(
                "Speech uses your browser's voices. Voices vary by browser and "
                "operating system."
            )
        return

    # Read-aloud reuses the shared voice assistant
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
# Indoor Navigation Demo - simulation on a fictional two-floor map
# ---------------------------------------------------------------------------
NAV_CLICK_MODES = ("Set start", "Set destination", "Close/open segment")


def _nav_label(graph, node_id: str) -> str:
    return f"{graph.display_name(node_id)} · {nav_floor_label(graph.floor(node_id))}"


def _nav_speak(text: str) -> None:
    if runtime.is_hosted():
        browser_speech.speak(text, st.session_state.get("speech_rate", vas.DEFAULT_RATE))
    elif vas.is_available():
        vas.assistant.speak(text)


def _nav_set_profile(profile_key: str) -> None:
    st.session_state[NAV_PROFILE] = profile_key
    st.session_state[NAV_PLAYING] = False


def _render_nav_guide_panel(
    result: nav.RouteResult,
    baseline: nav.RouteResult | None,
    steps: list,
    current_index: int,
    profile_key: str,
) -> None:
    """Keep the walk-through and its controls next to the map."""
    with st.container(border=True):
        st.subheader("Walk-through")
        st.caption(
            "Simulated step by step on the fictional map. Text always "
            "matches the spoken version."
        )

        if result.error:
            st.error(result.error)
            return

        if result.unreachable:
            st.warning(
                result.unreachable_reason
                or "No route exists under the current simulated constraints."
            )
            if (
                profile_key != "fastest"
                and baseline is not None
                and baseline.found
            ):
                st.caption(
                    "The Fastest profile still has a route, but it may use "
                    "stairs or crowded segments."
                )
                st.button(
                    "Switch to Fastest profile",
                    key="nav_use_fastest",
                    on_click=_nav_set_profile,
                    args=("fastest",),
                    width="stretch",
                )
            else:
                st.caption(
                    "Choose another profile, reopen a segment, or pick a "
                    "different destination."
                )
            return

        step = steps[current_index]
        st.markdown(
            f"**{result.cost:.0f} m · {nav_steps.format_duration(result.eta_s)}"
            f" · {result.stairs_count} stairs · {result.lift_count} lift uses**"
        )
        st.caption(
            f"Step {current_index + 1} of {len(steps)} · "
            f"{nav_floor_label(step.floor)}"
        )
        st.markdown(f"#### {step.instruction}")

        next_leg = (
            nav_steps.format_distance(step.distance_m)
            if step.action not in {"lift", "stairs", "arrive"}
            else "floor change"
            if step.action in {"lift", "stairs"}
            else "arrived"
        )
        remaining_s = max(0, result.eta_s - step.eta_s)
        st.caption(
            f"Next leg {next_leg} · {nav_steps.format_distance(step.remaining_m)}"
            f" left · about {nav_steps.format_duration(remaining_s)} remaining"
        )

        def _pause() -> None:
            st.session_state[NAV_PLAYING] = False

        def _step_by(delta: int) -> None:
            count = int(st.session_state.get(NAV_STEP_COUNT, 1))
            current = int(st.session_state.get(NAV_STEP, 0))
            st.session_state[NAV_STEP] = min(max(0, current + delta), count - 1)
            st.session_state[NAV_PLAYING] = False

        def _toggle_play() -> None:
            playing = not st.session_state.get(NAV_PLAYING, False)
            count = int(st.session_state.get(NAV_STEP_COUNT, 1))
            if playing and int(st.session_state.get(NAV_STEP, 0)) >= count - 1:
                st.session_state[NAV_STEP] = 0
            st.session_state[NAV_PLAYING] = playing
            st.session_state[NAV_LAST_ADVANCE] = time.monotonic()

        if len(steps) > 1:
            st.slider(
                "Route progress",
                0,
                len(steps) - 1,
                key=NAV_STEP,
                on_change=_pause,
                help="Drag to jump to any step. Playing advances every few "
                "seconds and pauses when you adjust the slider.",
            )

        playing = st.session_state.get(NAV_PLAYING, False)
        nav_prev, nav_play, nav_next = st.columns(3)
        nav_prev.button(
            "Previous",
            key="nav_prev_step",
            disabled=current_index == 0,
            on_click=_step_by,
            args=(-1,),
            width="stretch",
        )
        nav_play.button(
            "Pause" if playing else "Play",
            key="nav_play",
            type="primary",
            disabled=len(steps) <= 1,
            on_click=_toggle_play,
            width="stretch",
        )
        nav_next.button(
            "Next",
            key="nav_next_step",
            disabled=current_index >= len(steps) - 1,
            on_click=_step_by,
            args=(1,),
            width="stretch",
        )

        speak_auto = st.checkbox(
            "Speak each step automatically",
            key=NAV_SPEAK_STEPS,
            help="Speech is optional; every instruction is always shown as text.",
        )
        if st.button("Speak this step", key="nav_speak_step", width="stretch"):
            _nav_speak(step.spoken)
            st.session_state[NAV_LAST_SPOKEN] = current_index

        if st.session_state.get(NAV_PLAYING) and len(steps) > 1:
            interval = 2.6 if speak_auto else 1.8

            @st.fragment(run_every=interval)
            def _nav_autoplay() -> None:
                now = time.monotonic()
                last = float(st.session_state.get(NAV_LAST_ADVANCE, 0.0))
                if now - last < interval * 0.8:
                    return
                index = int(st.session_state.get(NAV_STEP, 0))
                if index + 1 < len(steps):
                    st.session_state[NAV_STEP] = index + 1
                    st.session_state[NAV_LAST_ADVANCE] = now
                    st.rerun()
                else:
                    st.session_state[NAV_PLAYING] = False
                    st.rerun()

            _nav_autoplay()

        if speak_auto and st.session_state.get(NAV_LAST_SPOKEN) != current_index:
            _nav_speak(step.spoken)
            st.session_state[NAV_LAST_SPOKEN] = current_index


def _render_nav_route_details(
    result: nav.RouteResult,
    baseline: nav.RouteResult | None,
    steps: list,
    profile_key: str,
) -> None:
    if not result.found:
        return
    with st.expander("Route details, text instructions, and comparisons"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Distance", f"{result.cost:.0f} m")
        m2.metric("Estimated time", nav_steps.format_duration(result.eta_s))
        m3.metric("Stairs", str(result.stairs_count))
        m4.metric("Lift uses", str(result.lift_count))

        details = []
        if result.min_width_m is not None:
            details.append(f"narrowest simulated segment {result.min_width_m:.1f} m")
        if result.detour_m:
            details.append(f"{result.detour_m:.0f} m longer than the fastest route")
        if result.constraints:
            details.append(" · ".join(result.constraints))
        if details:
            st.caption("; ".join(details).capitalize() + ".")

        if result.explanation:
            st.info(result.explanation)

        if baseline is not None and baseline.found and profile_key != "fastest":
            st.markdown("**Profile comparison**")
            st.dataframe(
                [
                    {
                        "Profile": result.profile_label,
                        "Distance (m)": round(result.cost, 1),
                        "ETA": nav_steps.format_duration(result.eta_s),
                        "Stairs": result.stairs_count,
                        "Lift uses": result.lift_count,
                    },
                    {
                        "Profile": "Fastest",
                        "Distance (m)": round(baseline.cost, 1),
                        "ETA": nav_steps.format_duration(baseline.eta_s),
                        "Stairs": baseline.stairs_count,
                        "Lift uses": baseline.lift_count,
                    },
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Both rows describe the same fictional map. ETA includes "
                "fixed simulated penalties for stairs, lifts, and doors."
            )

        st.markdown("**Full route**")
        st.markdown(" → ".join(f"**{name}**" for name in result.path_display))

        st.markdown("**Turn-by-turn text**")
        for step in steps:
            st.markdown(f"**{step.index + 1}. {step.name}** - {step.instruction}")

        st.caption(
            f"Nodes explored ({result.explored_count}): "
            + " → ".join(result.explored_names)
        )


def _render_nav_algorithm_maps(
    graph, start: str, dest: str, blocked_edges, blocked_nodes, speed: float
) -> None:
    """Show the concrete BFS and A* paths side by side as route maps."""
    st.subheader("BFS vs A* route maps")
    st.caption(
        "Both maps use the Fastest profile because BFS ignores edge weights. "
        "The active simulated closures are included."
    )
    bfs_result = nav.plan_route(
        graph,
        start,
        dest,
        profile_key="fastest",
        blocked_edges=blocked_edges,
        blocked_nodes=blocked_nodes,
        algorithm="BFS",
        speed_mps=speed,
        include_baseline=False,
    )
    astar_result = nav.plan_route(
        graph,
        start,
        dest,
        profile_key="fastest",
        blocked_edges=blocked_edges,
        blocked_nodes=blocked_nodes,
        algorithm="A*",
        speed_mps=speed,
        include_baseline=False,
    )
    if not (bfs_result.found and astar_result.found):
        st.info(
            "BFS and A* need an open route between the selected locations "
            "under the current simulated closures."
        )
        return

    same_path = bfs_result.path == astar_result.path
    map_left, map_right = st.columns(2)
    for col, label, result, colour, key in (
        (map_left, "BFS", bfs_result, "#B45309", "nav_algo_bfs"),
        (map_right, "A*", astar_result, map_view.ROUTE, "nav_algo_astar"),
    ):
        with col:
            st.markdown(f"**{label}**")
            st.caption(
                f"{result.hops} hops · {result.cost:.1f} m · "
                f"{result.explored_count} nodes explored · "
                f"{result.time_ms} ms"
            )
            figure = map_view.build_route_figure(
                graph,
                path=result.path,
                start=start,
                goal=dest,
                blocked_edges=blocked_edges,
                blocked_nodes=blocked_nodes,
                label_scope="route",
                route_color=colour,
                route_name=f"{label} path",
                height=520,
            )
            st.plotly_chart(
                figure,
                width="stretch",
                key=key,
                config={"displayModeBar": False, "scrollZoom": False},
            )
            st.caption("Path: " + " → ".join(result.path_display))

    st.markdown("**Algorithm totals**")
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

    if same_path:
        st.info(
            "Both algorithms chose the same path for this pair. Try "
            "Main Hall → Staff Room to see their paths diverge."
        )
    else:
        st.caption(
            "BFS minimizes hops; A* minimizes walking distance. Notice where "
            "the two coloured paths diverge."
        )


def render_navigation() -> None:
    components.page_header(
        "Indoor Navigation Demo",
        "Click the map, choose a profile, and walk the route without "
        "leaving the map view.",
    )
    components.safety_notice()  # mandatory on the navigation page
    st.caption(
        "Simulation only: a fixed fictional map, not live location or "
        "guidance. Step-free and quiet profiles are preferences on this "
        "sample graph, not accessibility certification."
    )

    graph = nav.get_graph()
    scenarios = nav_scenarios.all_scenarios(graph)
    node_ids = graph.ids()

    st.session_state.setdefault(NAV_START, "entrance")
    st.session_state.setdefault(
        NAV_DEST, "meeting_room" if "meeting_room" in node_ids else node_ids[-1]
    )
    st.session_state.setdefault(NAV_PROFILE, "fastest")
    st.session_state.setdefault(NAV_SPEED, 1.1)
    st.session_state.setdefault(NAV_SCENARIO, "none")
    st.session_state.setdefault(NAV_CLOSURES, set())
    st.session_state.setdefault(NAV_CLICK_MODE, NAV_CLICK_MODES[0])
    st.session_state.setdefault(NAV_MAP_EPOCH, 0)
    st.session_state.setdefault(NAV_STEP, 0)
    st.session_state.setdefault(NAV_PLAYING, False)
    st.session_state.setdefault(NAV_SPEAK_STEPS, False)
    st.session_state.setdefault(NAV_SHOW_EXPLORED, False)
    st.session_state.setdefault(NAV_SHOW_LABELS, True)
    st.session_state.setdefault(NAV_LAST_ADVANCE, 0.0)
    st.session_state.setdefault(NAV_LAST_SPOKEN, -1)

    if st.session_state[NAV_START] not in graph.nodes:
        st.session_state[NAV_START] = "entrance"
    if st.session_state[NAV_DEST] not in graph.nodes:
        st.session_state[NAV_DEST] = "meeting_room"
    if st.session_state[NAV_PROFILE] not in nav_profiles.PROFILES:
        st.session_state[NAV_PROFILE] = "fastest"
    if st.session_state[NAV_SCENARIO] not in scenarios:
        st.session_state[NAV_SCENARIO] = "none"
    if st.session_state[NAV_CLICK_MODE] not in NAV_CLICK_MODES:
        st.session_state[NAV_CLICK_MODE] = NAV_CLICK_MODES[0]

    # Apply a map click from the previous run before the widgets are created.
    pending = st.session_state.pop(NAV_PENDING, None)
    if pending:
        if pending.get("kind") == "node":
            target = NAV_START if pending.get("mode") == "Set start" else NAV_DEST
            node = pending.get("node")
            if node in graph.nodes:
                st.session_state[target] = node
                role = (
                    "Start"
                    if pending.get("mode") == "Set start"
                    else "Destination"
                )
                st.session_state[NAV_FLASH] = (
                    f"{role} set to {graph.display_name(node)}."
                )
        elif pending.get("kind") == "edge":
            try:
                a, b = sorted((pending["edge"][0], pending["edge"][1]))
            except (KeyError, IndexError, TypeError):
                a = b = ""
            if a in graph.nodes and b in graph.nodes and a != b:
                closures = set(st.session_state.get(NAV_CLOSURES, set()))
                key = (a, b)
                if key in closures:
                    closures.remove(key)
                    action = "reopened"
                else:
                    closures.add(key)
                    action = "closed"
                st.session_state[NAV_CLOSURES] = closures
                st.session_state[NAV_FLASH] = (
                    f"Segment {action}: {graph.display_name(a)} ↔ "
                    f"{graph.display_name(b)}."
                )
        st.session_state[NAV_PLAYING] = False

    flash = st.session_state.pop(NAV_FLASH, None)
    if flash:
        st.toast(flash)

    with st.container(border=True):
        st.markdown("#### Plan a route")
        s1, s2, s3 = st.columns([1.2, 1.2, 1])
        start = s1.selectbox(
            "Start location",
            node_ids,
            key=NAV_START,
            format_func=lambda node: _nav_label(graph, node),
        )
        dest = s2.selectbox(
            "Destination",
            node_ids,
            key=NAV_DEST,
            format_func=lambda node: _nav_label(graph, node),
        )
        scenario_key = s3.selectbox(
            "Closure scenario",
            list(scenarios),
            format_func=lambda key: scenarios[key].label,
            key=NAV_SCENARIO,
            help=scenarios[
                st.session_state.get(NAV_SCENARIO, "none")
            ].description,
        )
        profile_key = st.radio(
            "Route profile",
            list(nav_profiles.PROFILE_ORDER),
            format_func=lambda key: nav_profiles.PROFILES[key].label,
            horizontal=True,
            key=NAV_PROFILE,
            help="Fastest = shortest distance. Step-free = no stairs, lift "
            "required for floor changes. Quiet = penalise crowded segments.",
        )
        st.caption(nav_profiles.PROFILES[profile_key].description)

        with st.expander("Advanced options"):
            a1, a2 = st.columns(2)
            speed = a1.slider(
                "Walking speed",
                min_value=0.8,
                max_value=1.4,
                step=0.1,
                key=NAV_SPEED,
                format="%.1f m/s",
                help="Used only to estimate time on the fictional map.",
            )
            show_explored = a2.checkbox(
                "Show explored nodes",
                key=NAV_SHOW_EXPLORED,
                help="Highlights nodes the selected search visited, for "
                "algorithm comparison.",
            )
            show_labels = a2.checkbox(
                "Show location names",
                key=NAV_SHOW_LABELS,
                help="Turn names off to reduce clutter; hover any marker to "
                "read its full name.",
            )

    def _reset_nav_closures() -> None:
        st.session_state[NAV_SCENARIO] = "none"
        st.session_state[NAV_CLOSURES] = set()
        st.session_state[NAV_PLAYING] = False

    custom_closures = set(st.session_state.get(NAV_CLOSURES, set()))
    effective_edges = frozenset(scenarios[scenario_key].blocked_edges) | frozenset(
        custom_closures
    )
    effective_nodes = frozenset(scenarios[scenario_key].blocked_nodes)
    closure_labels = nav_profiles.blocked_summary(
        graph, effective_edges, effective_nodes
    )

    result = nav.plan_route(
        graph,
        start,
        dest,
        profile_key=profile_key,
        blocked_edges=effective_edges,
        blocked_nodes=effective_nodes,
        speed_mps=speed,
    )
    baseline = None
    if profile_key != "fastest":
        baseline = nav.plan_route(
            graph,
            start,
            dest,
            profile_key="fastest",
            blocked_edges=effective_edges,
            blocked_nodes=effective_nodes,
            speed_mps=speed,
            include_baseline=False,
        )

    steps = (
        nav_steps.build_steps(graph, result.path, speed) if result.found else []
    )
    signature = (
        start,
        dest,
        profile_key,
        round(speed, 2),
        tuple(sorted(effective_edges)),
        tuple(sorted(effective_nodes)),
        tuple(result.path),
    )
    if st.session_state.get(NAV_STEP_SIG) != signature:
        st.session_state[NAV_STEP] = 0
        st.session_state[NAV_PLAYING] = False
        st.session_state[NAV_LAST_SPOKEN] = -1
        st.session_state[NAV_STEP_SIG] = signature

    st.session_state[NAV_STEP_COUNT] = len(steps)
    if steps:
        current_index = min(
            max(int(st.session_state.get(NAV_STEP, 0)), 0), len(steps) - 1
        )
        st.session_state[NAV_STEP] = current_index
        current_node = steps[current_index].node_id
    else:
        current_index = 0
        current_node = None

    map_col, guide_col = st.columns([1.65, 1], gap="large")

    with map_col:
        toolbar_left, toolbar_right = st.columns([3.4, 1])
        click_mode = toolbar_left.radio(
            "Map click action",
            NAV_CLICK_MODES,
            horizontal=True,
            key=NAV_CLICK_MODE,
            help="Choose whether clicking a location sets the start, sets "
            "the destination, or closes/reopens a segment.",
        )
        toolbar_right.button(
            "Reset closures",
            key="nav_reset_closures",
            on_click=_reset_nav_closures,
            disabled=not (effective_edges or effective_nodes),
            width="stretch",
        )
        if closure_labels:
            st.caption("Closed: " + "; ".join(closure_labels))
        elif click_mode == "Close/open segment":
            st.caption(
                "No simulated closures. Click a small square handle on a "
                "segment to close it."
            )
        else:
            st.caption("No simulated closures.")
        st.caption(
            "Click a location or segment; the route and walk-through update "
            "automatically."
        )

        figure = map_view.build_route_figure(
            graph,
            path=result.path if result.found else (),
            start=start,
            goal=dest,
            current=(
                current_node
                if current_node
                and (current_index > 0 or st.session_state.get(NAV_PLAYING))
                else None
            ),
            explored=(
                result.explored_ids if show_explored and result.found else ()
            ),
            blocked_edges=effective_edges,
            blocked_nodes=effective_nodes,
            show_closures=click_mode == "Close/open segment",
            label_scope="all" if show_labels else "none",
        )
        map_epoch = int(st.session_state.get(NAV_MAP_EPOCH, 0))
        event = st.plotly_chart(
            figure,
            width="stretch",
            key=f"nav_map_{map_epoch}",
            on_select="rerun",
            selection_mode="points",
            config={"displayModeBar": False, "scrollZoom": False},
        )
        selected = map_view.decode_selection(event)
        if selected:
            if selected[0] == "node" and click_mode in {
                "Set start",
                "Set destination",
            }:
                st.session_state[NAV_PENDING] = {
                    "kind": "node",
                    "mode": click_mode,
                    "node": selected[1],
                }
                st.session_state[NAV_MAP_EPOCH] = map_epoch + 1
                st.rerun()
            elif selected[0] == "edge" and click_mode == "Close/open segment":
                st.session_state[NAV_PENDING] = {
                    "kind": "edge",
                    "edge": (selected[1], selected[2]),
                }
                st.session_state[NAV_MAP_EPOCH] = map_epoch + 1
                st.rerun()

    with guide_col:
        _render_nav_guide_panel(
            result, baseline, steps, current_index, profile_key
        )

    _render_nav_route_details(result, baseline, steps, profile_key)
    _render_nav_algorithm_maps(
        graph, start, dest, effective_edges, effective_nodes, speed
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
        "change it on the Scene Assistant page (Image upload mode). "
        "Open-vocabulary mode keeps its own lower threshold."
    )

    st.subheader("Model / device status")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            "- **Device:** CPU only (GPU optional, not required)\n"
            "- **Detection models:** yolo11n (80 classes) and YOLO-World "
            "(open vocabulary), Ultralytics, AGPL-3.0; cached locally after "
            "first use\n"
            "- **Depth model:** Depth Anything V2 Small (relative bands only)\n"
            + (
                "- **Speech:** browser Web Speech API (hosted build)\n"
                if runtime.is_hosted()
                else "- **Speech:** eSpeak NG offline text-to-speech\n"
            )
            + "- **Navigation:** hand-written BFS and A* with Fastest, "
            "Step-free, and Quiet profiles on a fixed fictional map"
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
        "Every image you upload is processed in memory by the app. It never "
        "saves, sells, or shares your images, OCR text, or voice data. There "
        "are no accounts, analytics, or telemetry. On a public deployment, "
        "processing happens on the hosting server - do not upload sensitive "
        "images to a shared instance."
    )

    st.subheader("Known limitations")
    st.markdown(
        "- Educational prototype only; not a mobility or safety device.\n"
        "- Detection recognizes only classes in the model's training labels; "
        "open-vocabulary prompts may miss objects or produce false positives.\n"
        "- Proximity bands are relative within one image, never metres, and "
        "must not be used to judge real-world distance or safety.\n"
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
    # camera (easy reversal + guaranteed release), and any
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
