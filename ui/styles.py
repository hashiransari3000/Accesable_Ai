"""Design tokens and global CSS for AccessAble AI.

Centralizes the visual identity so every page stays consistent
(HCI golden rule: consistency). Tokens are plain constants; the CSS is
injected once by the app shell.
"""

# ---- Colour tokens (light theme) ------------------------------------------
TEXT_PRIMARY = "#1F2933"      # deep navy/charcoal
TEXT_SECONDARY = "#52606D"    # slate grey
BG_PAGE = "#F5F7FA"           # soft neutral background
SURFACE = "#FFFFFF"           # cards / panels
ACCENT = "#0E7C86"            # restrained teal
ACCENT_DARK = "#0A5C63"
SUCCESS = "#1B7A43"
WARNING = "#B45309"
ERROR = "#B3261E"
BORDER = "#CBD2D9"

# ---- Spacing / radius scale -----------------------------------------------
RADIUS = "10px"

FONT_STACK = (
    "'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif"
)


def inject_global_css() -> None:
    """Apply the app-wide design system to the current Streamlit page."""
    st_css = f"""
    <style>
    /* Page background and base typography */
    .stApp {{
        background-color: {BG_PAGE};
        color: {TEXT_PRIMARY};
        font-family: {FONT_STACK};
    }}

    /* Sidebar: calm, high-contrast surface */
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.98rem;
        padding: 4px 2px;
    }}

    /* Headings */
    h1, h2, h3 {{
        color: {TEXT_PRIMARY};
        letter-spacing: -0.01em;
    }}

    /* Primary button: the one obvious action */
    .stButton > button[kind="primary"] {{
        background-color: {ACCENT};
        border-color: {ACCENT};
        color: #FFFFFF;
        border-radius: {RADIUS};
        font-weight: 600;
        padding: 0.55rem 1.1rem;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {ACCENT_DARK};
        border-color: {ACCENT_DARK};
    }}
    .stButton > button[kind="primary"]:focus-visible {{
        outline: 3px solid {ACCENT_DARK};
        outline-offset: 2px;
    }}

    /* Secondary buttons */
    .stButton > button {{
        border-radius: {RADIUS};
        color: {TEXT_PRIMARY};
    }}

    /* Download/upload controls: keep them tidy */
    .stDownloadButton > button {{
        border-radius: {RADIUS};
    }}

    /* Image container: soft border so previews read as content, not chrome */
    .stImage img, [data-testid="stImage"] {{
        border-radius: {RADIUS};
        border: 1px solid {BORDER};
    }}

    /* Reduce visual noise from default streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* Force comfortable line spacing for body text */
    .stMarkdown p {{ line-height: 1.6; }}

    /* Small helper class for muted text */
    .aa-muted {{
        color: {TEXT_SECONDARY};
        font-size: 0.92rem;
    }}
    </style>
    """
    import streamlit as st

    st.markdown(st_css, unsafe_allow_html=True)
