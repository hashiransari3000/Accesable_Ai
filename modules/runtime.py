"""Runtime environment detection.

On a hosted deployment the app cannot use the server's speakers or a
server-attached webcam, so the UI switches to browser-based speech and a
browser camera snapshot mode. Local runs keep the full offline feature set.

Detection is per request:
  - an explicit hosting flag always wins;
  - Streamlit Community Cloud / Spaces environment markers are honoured;
  - any browser URL that is not localhost is treated as remote, so phone
    and LAN access also get browser speech and the browser camera.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_remote_url(url: str) -> bool:
    """True when a browser URL is not a loopback/localhost address."""
    if not url:
        return False
    return (urlparse(url).hostname or "") not in _LOCAL_HOSTS


def is_hosted() -> bool:
    """True when the visitor is not running the app on their own machine."""
    if os.environ.get("ACCESSABLE_HOSTED", "").lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"):
        return True
    if os.environ.get("IS_STREAMLIT_CLOUD"):
        return True
    if os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit":
        return True

    try:
        import streamlit as st

        url = st.context.url or ""
    except Exception:
        return False
    return is_remote_url(url)
