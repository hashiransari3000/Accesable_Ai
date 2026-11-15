"""Design system for AccessAble AI.

Visual direction (grounded in the ui-ux-pro-max skill catalogs):
  - Typeface: Atkinson Hyperlegible (Braille Institute) - designed for
    low-vision readers; falls back to system fonts if offline.
  - High-contrast calm palette: near-black ink on soft off-white,
    deep-teal accent (5.7:1 on white), dark slate sidebar shell.
  - 16px+ base text, generous line height, 12px radius surfaces with
    subtle borders and soft shadows.
  - Focus: 3px visible ring with offset on every interactive element.
  - Motion: subtle 150-250ms transitions, fully disabled under
    prefers-reduced-motion.
  - Status is never colour-only: every banner pairs icon + text.
"""

# ---- Colour tokens ---------------------------------------------------------
INK = "#101828"           # primary text (17:1 on bg)
INK_SOFT = "#475569"      # secondary text (7.4:1 on bg)
BG_PAGE = "#F8FAFC"       # soft off-white
SURFACE = "#FFFFFF"
SIDEBAR_BG = "#0F172A"    # dark slate shell
SIDEBAR_TEXT = "#E2E8F0"
SIDEBAR_MUTED = "#94A3B8"
ACCENT = "#0F766E"        # deep teal (5.7:1 on white)
ACCENT_STRONG = "#115E59" # hover state
ACCENT_SOFT = "#F0FDFA"   # tinted background
ACCENT_BORDER = "#99F6E4"
SUCCESS = "#166534"
WARNING = "#92400E"
DANGER = "#B91C1C"
BORDER = "#E2E8F0"
RING = "#0D9488"          # focus ring colour (3:1+ vs page bg)

RADIUS = "12px"
FONT_STACK = (
    "'Atkinson Hyperlegible', 'Segoe UI', 'Inter', 'Helvetica Neue', "
    "Arial, sans-serif"
)


def inject_global_css() -> None:
    """Apply the app-wide design system to the current Streamlit page."""
    import streamlit as st

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');

        /* ---------- Base ---------- */
        .stApp {{
            background-color: {BG_PAGE};
            color: {INK};
            font-family: {FONT_STACK};
            font-size: 16px;
            line-height: 1.6;
        }}
        .stMarkdown p {{ line-height: 1.65; }}
        h1, h2, h3 {{ color: {INK}; letter-spacing: -0.015em; line-height: 1.25; }}
        h1 {{ font-size: 1.9rem; }}
        h2 {{ font-size: 1.35rem; }}
        h3 {{ font-size: 1.08rem; }}

        a, .stMarkdown a {{ color: {ACCENT}; text-decoration: underline; }}

        /* ---------- Dark sidebar shell ---------- */
        section[data-testid="stSidebar"] {{
            background-color: {SIDEBAR_BG};
            border-right: 1px solid {SIDEBAR_BG};
        }}
        section[data-testid="stSidebar"] * {{
            color: {SIDEBAR_TEXT};
        }}
        section[data-testid="stSidebar"] .stMarkdown {{
            font-family: {FONT_STACK};
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: #1E293B;
        }}
        /* App name in the sidebar */
        section[data-testid="stSidebar"] .aa-appname {{
            color: #FFFFFF;
            font-weight: 700;
            font-size: 1.25rem;
            letter-spacing: -0.01em;
        }}
        section[data-testid="stSidebar"] .aa-apptag {{
            color: {SIDEBAR_MUTED};
            font-size: 0.85rem;
        }}
        /* Nav list: radio rendered as a clean vertical menu with an
           active-item highlight (HCI: visibility of system status) */
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 9px 12px;
            margin: 2px 0;
            cursor: pointer;
            transition: background-color 180ms ease, border-color 180ms ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: #1E293B;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
            background: {ACCENT_STRONG};
            border-color: #134E4A;
            border-left: 3px solid #5EEAD4;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {{
            color: #FFFFFF;
            font-weight: 700;
        }}

        /* ---------- Page-level choice groups (mode / algorithm) ---------- */
        section[data-testid="stMain"] div[role="radiogroup"] {{
            gap: 8px;
        }}
        section[data-testid="stMain"] div[role="radiogroup"] label {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 999px;
            padding: 6px 16px;
            cursor: pointer;
            transition: background-color 160ms ease, border-color 160ms ease;
        }}
        section[data-testid="stMain"] div[role="radiogroup"] label:hover {{
            border-color: {ACCENT};
        }}
        section[data-testid="stMain"] div[role="radiogroup"] label:has(input:checked) {{
            background: {ACCENT_SOFT};
            border-color: {ACCENT_BORDER};
        }}
        section[data-testid="stMain"] div[role="radiogroup"] label:has(input:checked) p {{
            font-weight: 700;
            color: {ACCENT_STRONG};
        }}

        /* ---------- Focus visibility (WCAG 2.2) ---------- */
        .stApp *:focus-visible,
        .stApp button:focus-visible,
        .stApp [role="button"]:focus-visible,
        .stApp a:focus-visible,
        .stApp input:focus-visible,
        .stApp [data-baseweb="slider"] :focus-visible {{
            outline: 3px solid {RING};
            outline-offset: 2px;
            border-radius: 6px;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button {{
            border-radius: 10px;
            font-family: {FONT_STACK};
            font-weight: 600;
            padding: 0.55rem 1.15rem;
            border: 1px solid {BORDER};
            color: {INK};
            background: {SURFACE};
            cursor: pointer;
            transition: background-color 160ms ease, border-color 160ms ease,
                        transform 160ms ease, box-shadow 160ms ease;
        }}
        .stButton > button:hover {{
            border-color: {ACCENT};
            color: {ACCENT_STRONG};
        }}
        .stButton > button[kind="primary"] {{
            background-color: {ACCENT};
            border-color: {ACCENT};
            color: #FFFFFF;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {ACCENT_STRONG};
            border-color: {ACCENT_STRONG};
            color: #FFFFFF;
            box-shadow: 0 2px 10px rgba(15, 118, 110, 0.28);
        }}
        .stButton > button[kind="primary"]:active {{
            transform: translateY(1px);
        }}
        .stButton > button:disabled {{
            opacity: 0.55;
            cursor: not-allowed;
        }}

        /* ---------- Cards / containers ---------- */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {BORDER};
            border-radius: {RADIUS};
            background-color: {SURFACE};
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
        }}

        /* ---------- Metric tiles ---------- */
        [data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: {RADIUS};
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
        }}
        [data-testid="stMetricValue"] {{
            color: {INK};
            font-weight: 700;
        }}
        [data-testid="stMetricLabel"] {{
            color: {INK_SOFT};
        }}

        /* ---------- Dataframe / tables ---------- */
        .stDataFrame {{
            border: 1px solid {BORDER};
            border-radius: {RADIUS};
            overflow: hidden;
        }}

        /* ---------- File uploader ---------- */
        [data-testid="stFileUploaderDropzone"] {{
            background: {SURFACE};
            border: 1.5px dashed #94A3B8;
            border-radius: {RADIUS};
            padding: 1.1rem 1.2rem;
            transition: border-color 160ms ease, background-color 160ms ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {ACCENT};
            background: {ACCENT_SOFT};
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            border-radius: 10px;
        }}

        /* ---------- Alerts / expanders ---------- */
        [data-testid="stAlert"] {{
            border-radius: 10px;
        }}
        [data-testid="stExpander"] details {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px;
            overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{
            font-weight: 600;
        }}

        /* ---------- Code / copy block ---------- */
        .stCode {{
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}

        /* ---------- Slider ---------- */
        [data-baseweb="slider"] > div {{
            color: {INK_SOFT};
        }}

        /* ---------- Muted helper text ---------- */
        .aa-muted {{
            color: {INK_SOFT};
            font-size: 0.92rem;
        }}

        /* ---------- Responsive tweaks ---------- */
        @media (max-width: 768px) {{
            h1 {{ font-size: 1.5rem; }}
            .stApp {{ font-size: 15px; }}
        }}

        /* ---------- Reduced motion (accessibility) ---------- */
        @media (prefers-reduced-motion: reduce) {{
            .stApp * {{
                transition: none !important;
                animation: none !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
