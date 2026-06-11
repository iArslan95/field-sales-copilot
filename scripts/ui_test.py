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
    assert len(at.chat_input) == 3, \
        f"expected 3 copilot inputs (main + 2 specialists), got {len(at.chat_input)}"
    assert len(at.button) >= 9, "suggestion chips should render on every panel"
    assert at.session_state["chat_main"], "briefing message should be seeded"
    first = at.session_state["chat_main"][0]
    assert first["role"] == "assistant" and "risk" in first["content"]

    # An outlet pick in the explorer must not crash the app.
    at.selectbox[0].select(at.selectbox[0].options[1])
    at.run()
    assert not at.exception, f"outlet explorer raised: {at.exception}"
    print("metrics:", " | ".join(f"{m.label}={m.value}" for m in at.metric))
    print("UI TEST OK")


if __name__ == "__main__":
    main()
