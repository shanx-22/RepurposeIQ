"""
utils/session_mgr.py
Feature 1 – Named project sessions: save/load st.session_state to JSON files.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Keys that are serialisable (exclude DataFrames, raw API caches, etc.)
SERIALISABLE_KEYS = {
    "drug", "category", "w_to", "w_np", "w_er", "conf_filter",
    "opp_cat_filter", "annotations", "disgenet_email", "disgenet_token",
    "comparison_mode", "drug_b", "custom_subgroups", "risk_dev_cost",
    "risk_market_size", "risk_p_success", "selected_opp_key",
    "flagged_pairs", "project_name",
}


def list_sessions() -> List[str]:
    return sorted([f.stem for f in SESSIONS_DIR.glob("*.json")])


def save_session(project_name: str) -> str:
    """Serialise current session_state to JSON. Returns timestamp string."""
    payload: Dict = {}
    for key in SERIALISABLE_KEYS:
        val = st.session_state.get(key)
        if val is not None:
            try:
                json.dumps(val)   # test serialisability
                payload[key] = val
            except (TypeError, ValueError):
                pass
    # Also persist the opp_df as records if present
    opp_df = st.session_state.get("opp_df")
    if opp_df is not None:
        try:
            # trials column contains lists — safe
            records = opp_df.to_dict(orient="records")
            payload["opp_df_records"] = records
        except Exception:
            pass
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["_saved_at"] = ts
    path = SESSIONS_DIR / f"{project_name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return ts


def load_session(project_name: str) -> Optional[str]:
    """Load JSON back into session_state. Returns saved_at timestamp or None."""
    path = SESSIONS_DIR / f"{project_name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    import pandas as pd
    for key, val in payload.items():
        if key == "opp_df_records":
            try:
                df = pd.DataFrame(val)
                # restore list columns
                for col in ["trials", "disease_genes"]:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
                st.session_state["opp_df"] = df
            except Exception:
                pass
        elif key != "_saved_at":
            st.session_state[key] = val
    return payload.get("_saved_at")


def render_session_widget():
    """
    Render the session save/load widget in the sidebar.
    Call this at the top of every page's sidebar block.
    """
    with st.sidebar:
        with st.expander("💾 Project sessions", expanded=False):
            # ── Save
            proj_name = st.text_input(
                "Project name",
                value=st.session_state.get("project_name", "my_project"),
                key="proj_name_input")
            if st.button("Save session", key="save_sess_btn", use_container_width=True):
                st.session_state["project_name"] = proj_name
                ts = save_session(proj_name)
                st.session_state["_last_saved"] = ts
                st.success(f"Saved ✓")
            if st.session_state.get("_last_saved"):
                st.caption(f"Last saved: {st.session_state['_last_saved']}")

            # ── Load
            sessions = list_sessions()
            if sessions:
                chosen = st.selectbox("Load session", ["— select —"] + sessions, key="load_sess_sel")
                if st.button("Load", key="load_sess_btn", use_container_width=True):
                    if chosen != "— select —":
                        ts = load_session(chosen)
                        if ts:
                            st.success(f"Loaded '{chosen}' (saved {ts})")
                            st.rerun()
                        else:
                            st.error("Session file not found.")
            else:
                st.caption("No saved sessions yet.")
