"""Reusable UI components for AccessAble AI.

Each component pairs colour with text (never colour alone) and keeps
essential information visible to reduce memory load. Icons are inline
SVGs (Lucide-style paths) always accompanied by visible text labels.
"""

import streamlit as st

from ui.styles import ACCENT_BORDER, ACCENT_SOFT, BORDER, SURFACE

# ---- Inline SVG icons (stroke-based, 24x24 viewBox) -------------------------
_ICONS = {
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
    "text": '<path d="M4 6h16M4 12h16M4 18h10"/>',
    "speaker": '<path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/>',
    "map": '<path d="M9 3 3 5v16l6-2 6 2 6-2V3l-6 2-6-2z"/><path d="M9 3v16M15 5v16"/>',
    "shield": '<path d="M12 2 4 5v6c0 5 3.4 9.7 8 11 4.6-1.3 8-6 8-11V5l-8-3z"/>',
    "camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
}

_COLOURS = {
    "teal": "#0F766E",
    "ink": "#101828",
    "white": "#FFFFFF",
}


def svg_icon(name: str, colour: str = "teal", size: int = 20) -> str:
    """Return an inline SVG (stroke style) as an HTML string."""
    path = _ICONS.get(name)
    if path is None:
        return ""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{_COLOURS.get(colour, colour)}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'role="img" aria-hidden="true">{path}</svg>'
    )


def page_header(title: str, description: str) -> None:
    """Consistent page header: small eyebrow + title + one-line description."""
    st.markdown(
        '<div style="margin-bottom:0.4rem;"><span class="aa-muted" '
        'style="text-transform:uppercase;letter-spacing:0.08em;'
        'font-size:0.78rem;font-weight:700;">AccessAble AI</span></div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption(description)


def status_banner(kind: str, message: str) -> None:
    """Show a status message. kind: success | warning | error | info.

    Streamlit banners ship their own icon, so state is never communicated
    by colour alone.
    """
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


def safety_notice() -> None:
    """The mandatory visible disclaimer, shown on Dashboard + Navigation."""
    st.warning(
        "**Educational prototype only.** AI may miss or misidentify objects. "
        "Do not rely on this app for safe mobility or obstacle avoidance."
    )


def privacy_notice() -> None:
    """Short, always-visible privacy statement."""
    st.markdown(
        '<div class="aa-muted" style="display:flex;gap:7px;align-items:center;">'
        f'{svg_icon("shield", colour="#0F766E", size=15)}'
        "<span><strong>Privacy:</strong> your image is processed locally on "
        "this machine. Nothing is uploaded, stored, or transmitted.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def feature_card(icon_name: str, title: str, body: str) -> None:
    """Professional feature card: SVG icon + title + body text."""
    icon_html = (
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'margin-bottom:6px;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:38px;height:38px;border-radius:10px;background:{ACCENT_SOFT};'
        f'border:1px solid {ACCENT_BORDER};">{svg_icon(icon_name)}</span>'
        f'<span style="font-weight:700;font-size:1.02rem;">{title}</span>'
        f"</div>"
    )
    with st.container(border=True):
        st.markdown(icon_html, unsafe_allow_html=True)
        st.markdown(f'<span class="aa-muted">{body}</span>', unsafe_allow_html=True)


def quick_start_steps() -> None:
    """Visible numbered instructions so new users never rely on memory."""
    st.markdown(
        '<div style="display:flex;flex-direction:column;gap:8px;">'
        '<div><strong>1.</strong> Open <strong>Scene Assistant</strong> in the sidebar.</div>'
        '<div><strong>2.</strong> Upload an image (JPG or PNG) or pick a sample.</div>'
        '<div><strong>3.</strong> Press <strong>Analyze image</strong> and read the results.</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def status_pill(label: str, value: str, ready: bool) -> None:
    """Small system-readiness pill: text + tinted dot, never colour-only."""
    dot = "#16A34A" if ready else "#94A3B8"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;'
        f'background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:999px;padding:6px 14px;width:fit-content;">'
        f'<span style="width:8px;height:8px;border-radius:50%;'
        f'background:{dot};display:inline-block;"></span>'
        f"<span><strong>{label}:</strong> {value}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
