"""Headless UI test via streamlit.testing: the app renders with KPIs, the
local briefing message, suggestion chips and a chat input — without any
network call (the agent only runs on user input).

    python scripts/ui_test.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    assert len(at.metric) == 4, f"expected 4 KPI metrics, got {len(at.metric)}"
    assert len(at.chat_input) == 1, "copilot chat input should render"
    assert len(at.button) >= 3, "suggestion chips should render"
    assert at.session_state["chat"], "briefing message should be seeded"
    first = at.session_state["chat"][0]
    assert first["role"] == "assistant" and "risk" in first["content"]
    print("metrics:", " | ".join(f"{m.label}={m.value}" for m in at.metric))
    print("UI TEST OK")


if __name__ == "__main__":
    main()
