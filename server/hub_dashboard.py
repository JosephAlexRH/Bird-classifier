"""Fill in the bird hub page's per-request values (remote mode and opening tab)."""
from __future__ import annotations

TABS = ("cameras", "fleet", "settings")


def hub_page(page_html: str, *, remote: bool, initial_tab: str = "cameras") -> str:
    """``initial_tab`` is cameras, fleet or settings; older callers' "live" means cameras."""
    tab = initial_tab if initial_tab in TABS else "cameras"
    return (page_html.replace("__REMOTE__", "true" if remote else "false")
            .replace("__INITIAL_TAB__", tab))
