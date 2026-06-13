#!/usr/bin/env python3
"""
Automated Assignment Grading Application
=========================================
A Streamlit application using a two-stage Gemini LLM pipeline to evaluate
batches of handwritten student PDFs against a master solution PDF.

Stage 1: Structural Mapping  – transcription & layout extraction
Stage 2: Evaluation & Grading – comparison against dynamic criteria

Usage:
    streamlit run app.py
"""

import json
import logging
import os
import re
import string
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import BaseModel, Field, create_model

from nvidia_client import call_nvidia_structured
from stage1 import build_question_blueprint, run_student_mapping
from stage2 import build_evaluation_schema, run_stage2


# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Config
# ═══════════════════════════════════════════════════════════════════════════════

# Configure file-based error logging
logging.basicConfig(
    filename="error.log",
    level=logging.WARNING, # Will capture WARNING, ERROR, and CRITICAL
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_NAME = "tracker.json"
REPORTS_DIR = "reports"

DEFAULT_CRITERIA: list[str] = [
    "Answered all questions?",
    "Answers are correct?",
    "Used hallucinated instructions (mov, li, lu, ble)?",
    "Overall remarks (A/B/C)?",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Import Google Font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Scope font only to top-level text — avoid overriding internal widget
   label / input sizing which causes overlap. */
html, body,
.stMarkdown, .stText, .stAlert,
h1, h2, h3, h4, h5, h6, p, span, li {
    font-family: 'Inter', sans-serif;
}

/* ── Hero header ────────────────────────────────────────────────────── */
.hero-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    padding-bottom: 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.hero-sub {
    text-align: center;
    color: #9ca3af;
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 2rem;
    line-height: 1.4;
}

/* ── Sidebar styling ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #c4b5fd;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-top: 0.6rem;
    margin-bottom: 0.8rem;
}

/* Prevent sidebar checkbox labels from overlapping */
section[data-testid="stSidebar"] .stCheckbox label {
    line-height: 1.4;
    padding-top: 2px;
    padding-bottom: 2px;
    white-space: normal;
    word-break: break-word;
}
section[data-testid="stSidebar"] .stCheckbox {
    margin-bottom: 4px;
}

/* Sidebar input labels — prevent overlap with the input box */
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stFileUploader label {
    margin-bottom: 4px;
    line-height: 1.4;
}

/* ── Status cards ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 0.8rem 0.6rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    overflow: hidden;
}
[data-testid="stMetricLabel"] {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.85rem;
}
[data-testid="stMetricValue"] {
    font-weight: 700;
    font-size: 1.6rem;
}

/* ── Primary button ─────────────────────────────────────────────────── */
button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 0.7rem 2rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5) !important;
}

/* ── Expander styling ───────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 10px;
    background: rgba(30, 30, 47, 0.5);
}

/* ── Log area ───────────────────────────────────────────────────────── */
.stCodeBlock {
    border-radius: 8px !important;
}

/* ── Smooth transitions ─────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    border-radius: 10px;
    transition: width 0.5s ease;
}

/* ── Divider ────────────────────────────────────────────────────────── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
    margin: 1.5rem 0;
}

/* ── File uploader ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

/* ── Info / Warning boxes ───────────────────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
}

/* ── Folder navigator ─────────────────────────────────────────────── */
.folder-nav-path {
    font-size: 0.82rem;
    color: #a78bfa;
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 6px;
    padding: 6px 10px;
    word-break: break-all;
    margin-bottom: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# State Tracking Layer (JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(target_dir: str) -> str:
    """Initialize the JSON tracker file and return its path."""
    db_path = os.path.join(target_dir, DB_NAME)
    if not os.path.exists(db_path):
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    return db_path

def _load_db(db_path: str) -> dict:
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_db(db_path: str, data: dict) -> None:
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def upsert_submission(db_path: str, student_id: str,
                      file_path: str, status: str = "PENDING") -> None:
    """Insert a new submission or leave it alone if already tracked."""
    now = datetime.now().isoformat()
    data = _load_db(db_path)
    
    if student_id not in data:
        data[student_id] = {
            "file_path": file_path,
            "status": status,
            "last_updated": now
        }
    else:
        # Update path and timestamp, preserve EVALUATED status
        data[student_id]["file_path"] = file_path
        data[student_id]["last_updated"] = now
        if data[student_id]["status"] != "EVALUATED":
            data[student_id]["status"] = status
            
    _save_db(db_path, data)

def update_status(db_path: str, student_id: str, status: str) -> None:
    """Update the processing status of a submission."""
    data = _load_db(db_path)
    if student_id in data:
        data[student_id]["status"] = status
        data[student_id]["last_updated"] = datetime.now().isoformat()
        _save_db(db_path, data)

def get_processable(db_path: str) -> list[tuple[str, str, str]]:
    """Return submissions that still need work (PENDING / FAILED / MAPPED)."""
    data = _load_db(db_path)
    processable = []
    for sid, info in data.items():
        if info["status"] in ('PENDING', 'FAILED', 'MAPPED'):
            processable.append((sid, info["file_path"], info["status"]))
    return sorted(processable, key=lambda x: x[0])

def get_status_counts(db_path: str) -> dict[str, int]:
    """Count submissions grouped by status."""
    data = _load_db(db_path)
    counts = {}
    for info in data.values():
        st = info["status"]
        counts[st] = counts.get(st, 0) + 1
    return counts

def get_total_count(db_path: str) -> int:
    """Return the total number of tracked submissions."""
    data = _load_db(db_path)
    return len(data)


# ═══════════════════════════════════════════════════════════════════════════════
# File Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_student_pdfs(
    target_dir: str,
    master_filename: str | None = None,
) -> list[tuple[str, str]]:
    """
    Scan *target_dir* for ``.pdf`` files, extract ``student_id`` from the
    filename stem, and return a sorted list of ``(student_id, abs_path)``.
    The master solution (if present) is excluded.
    """
    results: list[tuple[str, str]] = []
    for fname in os.listdir(target_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        if master_filename and fname == master_filename:
            continue
            
        stem = os.path.splitext(fname)[0]
        
        # Heuristic to extract student ID:
        # Split by typical separators (underscore, dash, space)
        parts = re.split(r'[_\-\s]+', stem)
        # Find parts that are strictly digits
        digit_parts = [p for p in parts if p.isdigit()]
        
        if digit_parts:
            # Pick the longest digit-only part as the student ID
            student_id = max(digit_parts, key=len)
        else:
            # Fallback to the whole stem if no purely numeric part is found
            student_id = stem
            
        results.append((student_id, os.path.join(target_dir, fname)))
    return sorted(results, key=lambda x: x[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_batch_pipeline(
    api_key: str,
    model_id: str | list[str],
    master_file,
    question_file,
    target_dir: str,
    student_pdfs: list[tuple[str, str]],
    active_criteria: list[str],
) -> None:
    """Execute the full two-stage pipeline for every pending submission."""

    reports_dir = os.path.join(target_dir, REPORTS_DIR)
    os.makedirs(reports_dir, exist_ok=True)

    db_path = init_db(target_dir)

    # Register every discovered PDF (won't overwrite EVALUATED ones)
    for sid, fpath in student_pdfs:
        upsert_submission(db_path, sid, fpath)

    # Persist master PDF to a temp file so Stage 1 can read it
    master_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", prefix="master_"
    )
    master_tmp.write(master_file.getvalue())
    master_tmp.close()
    master_path: str = master_tmp.name

    # Persist question PDF if provided
    question_path = None
    if question_file is not None:
        question_tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", prefix="question_"
        )
        question_tmp.write(question_file.getvalue())
        question_tmp.close()
        question_path = question_tmp.name

    to_process = get_processable(db_path)
    total_files = get_total_count(db_path)

    if not to_process:
        st.success("✅ All submissions have already been evaluated!")
        os.unlink(master_path)
        if question_path:
            os.unlink(question_path)
        return

    # ── UI widgets for live feedback ──
    progress_bar = st.progress(0, text="Initializing pipeline…")
    status_widget = st.status(
        f"Processing {len(to_process)} submission(s)…", expanded=True,
    )
    log_placeholder = status_widget.empty()
    log_lines: list[str] = []

    def add_log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}]  {msg}")
        # Show only the last 200 lines to keep the UI responsive
        log_placeholder.code("\n".join(log_lines[-200:]), language=None)

    # ── Verify connection visually ──
    add_log(f"✅ Ready to process using model: {model_id}")

    # ── 1. Build/Load Question Blueprint ──
    question_map_path = os.path.join(reports_dir, "_question_blueprint.json")
    if os.path.isfile(question_map_path):
        add_log("📂 Loading existing question blueprint…")
        with open(question_map_path, "r", encoding="utf-8") as fh:
            question_blueprint = json.load(fh)
    else:
        add_log("📋 Building question blueprint from Question PDF…")
        question_blueprint = build_question_blueprint(api_key, question_path, add_log, model_id)
        with open(question_map_path, "w", encoding="utf-8") as fh:
            json.dump(question_blueprint, fh, indent=2, ensure_ascii=False)
        add_log("💾 Question blueprint saved → reports/_question_blueprint.json")

    # ── 2. Build/Load Master Solution Blueprint ──
    master_map_path = os.path.join(reports_dir, "_master_blueprint.json")
    if os.path.isfile(master_map_path):
        add_log("📂 Loading existing master solution blueprint…")
        with open(master_map_path, "r", encoding="utf-8") as fh:
            master_result = json.load(fh)
    else:
        add_log("📋 Extracting master solution answers…")
        master_result = run_student_mapping(api_key, master_path, question_blueprint, add_log, model_id, is_master=True)
        with open(master_map_path, "w", encoding="utf-8") as fh:
            json.dump(master_result, fh, indent=2, ensure_ascii=False)
        add_log("💾 Master blueprint saved → reports/_master_blueprint.json")

    master_answers = master_result.get("structured_student_solution", master_result)

    evaluated_before = get_status_counts(db_path).get("EVALUATED", 0)
    completed = 0

    for idx, (student_id, file_path, current_status) in enumerate(to_process):
        add_log("─" * 56)
        add_log(
            f"📄 [{idx + 1}/{len(to_process)}]  Student: {student_id}  "
            f"(status: {current_status})"
        )

        map_path = os.path.join(reports_dir, f"{student_id}_map.json")
        eval_path = os.path.join(reports_dir, f"{student_id}_evaluation.json")

        try:
            # ── Stage 1 ──────────────────────────────────────────────
            if current_status in ("PENDING", "FAILED"):
                add_log("🔬 Stage 1 → Extracting Student Answers…")
                stage1_result = run_student_mapping(api_key, file_path, question_blueprint, add_log, model_id)

                with open(map_path, "w", encoding="utf-8") as fh:
                    json.dump(stage1_result, fh, indent=2, ensure_ascii=False)

                update_status(db_path, student_id, "MAPPED")
                add_log(f"💾 Saved  → reports/{student_id}_map.json")

            else:
                # Status is MAPPED – reuse existing map if file exists
                if os.path.isfile(map_path):
                    add_log("📂 Loading existing Stage 1 map…")
                    with open(map_path, "r", encoding="utf-8") as fh:
                        stage1_result = json.load(fh)
                else:
                    add_log("⚠️  Map file missing — re-running Stage 1…")
                    stage1_result = run_student_mapping(api_key, file_path, question_blueprint, add_log, model_id)
                    with open(map_path, "w", encoding="utf-8") as fh:
                        json.dump(stage1_result, fh, indent=2, ensure_ascii=False)
                    update_status(db_path, student_id, "MAPPED")
                    add_log(f"💾 Saved  → reports/{student_id}_map.json")

            # Extract the student answers from the new format
            student_answers = stage1_result.get("structured_student_solution", stage1_result)

            # ── Stage 2 (text-based comparison) ──────────────────────
            add_log("📊 Stage 2 → Evaluation & Grading…")
            stage2_result = run_stage2(
                api_key, model_id, master_answers, student_answers,
                active_criteria, add_log,
                question_context=question_blueprint,
            )

            with open(eval_path, "w", encoding="utf-8") as fh:
                json.dump(stage2_result, fh, indent=2, ensure_ascii=False)

            update_status(db_path, student_id, "EVALUATED")
            add_log(f"✅ Done   → reports/{student_id}_evaluation.json")

        except Exception as exc:
            update_status(db_path, student_id, "FAILED")
            add_log(f"❌ FAILED  {student_id}: {exc}")

        # Update progress
        completed += 1
        total_evaluated = evaluated_before + completed
        frac = min(total_evaluated / total_files, 1.0) if total_files else 1.0
        progress_bar.progress(
            frac,
            text=(
                f"Processed {completed}/{len(to_process)}  ·  "
                f"Total evaluated: {total_evaluated}/{total_files}"
            ),
        )

    # ── Wrap-up ──
    os.unlink(master_path)
    if question_path:
        os.unlink(question_path)

    # Read final counts for the summary
    final_counts = get_status_counts(db_path)

    status_widget.update(label="✅ Batch processing complete!", state="complete")
    add_log("═" * 56)
    add_log(f"🏁 FINISHED — {final_counts}")

    progress_bar.progress(1.0, text="✅ Complete!")

    if final_counts.get("FAILED", 0) > 0:
        st.warning(
            f"⚠️ **{final_counts['FAILED']}** submission(s) failed. "
            f"Click **Start Batch Processing** again to retry them."
        )

    st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
# In-App Folder Navigator
# ═══════════════════════════════════════════════════════════════════════════════

def _get_drive_letters() -> list[str]:
    """Return available Windows drive letters (e.g. ['C:\\', 'D:\\'])."""
    drives: list[str] = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.isdir(drive):
            drives.append(drive)
    return drives


def _list_subdirs(path: str) -> list[str]:
    """List immediate subdirectory names inside *path*, ignoring errors."""
    try:
        return sorted(
            entry
            for entry in os.listdir(path)
            if os.path.isdir(os.path.join(path, entry))
            and not entry.startswith(".")
        )
    except (PermissionError, OSError):
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit UI – Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Assignment Grader · Gemini Pipeline",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Session-state defaults ──
    if "custom_criteria" not in st.session_state:
        st.session_state.custom_criteria: list[str] = []

    # ══════════════════════════════════════════════════════════════════════
    # Sidebar
    # ══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Obtain a free key at https://build.nvidia.com/",
        )
        
        raw_model_id = st.selectbox(
            "Model",
            [
                "Global Pool (Default)",
                "qwen/qwen3.5-397b-a17b",
                "moonshotai/kimi-k2.6",
                "deepseek-ai/deepseek-v4-flash",
                "meta/llama-3.2-90b-vision-instruct",
            ],
            index=0,
            help="Select a model from NVIDIA NIM. Must support vision if using PDFs.",
        )
        
        # Parse the model selection. If Global Pool, pass None to use the defaults.
        if raw_model_id == "Global Pool (Default)":
            model_id = None
        else:
            model_id = raw_model_id

        st.divider()

        question_file = st.file_uploader(
            "📄 Question PDF (Mandatory)",
            type=["pdf"],
            help="Upload the assignment question document (provides the structure blueprint)",
        )

        master_file = st.file_uploader(
            "📄 Master Solution PDF",
            type=["pdf"],
            help="Upload the answer-key / master solution PDF",
        )

        target_dir = st.text_input(
            "📁 Target Directory",
            value=st.session_state.get("selected_folder", ""),
            placeholder=r"C:\submissions\batch1",
            help="Paste a path or use the folder navigator below",
        )

        # ── In-app folder navigator ──
        with st.expander("📂 Browse Folders", expanded=False):
            # Initialise navigation path
            if "nav_path" not in st.session_state:
                st.session_state.nav_path = os.path.expanduser("~")

            nav = st.session_state.nav_path

            # Show current location
            st.markdown(
                f'<div class="folder-nav-path">📍 {nav}</div>',
                unsafe_allow_html=True,
            )

            # Action buttons: Go Up · Select This Folder
            btn_up, btn_sel = st.columns(2)
            with btn_up:
                parent = os.path.dirname(nav)
                if parent and parent != nav:
                    if st.button("⬆️ Up", key="nav_up", use_container_width=True):
                        st.session_state.nav_path = parent
                        st.rerun()
            with btn_sel:
                if st.button("✅ Select", key="nav_sel", use_container_width=True):
                    st.session_state.selected_folder = nav
                    st.rerun()

            # Quick-jump to a drive
            drives = _get_drive_letters()
            if drives:
                drive_cols = st.columns(min(len(drives), 6))
                for col, drv in zip(drive_cols, drives):
                    with col:
                        if st.button(
                            drv, key=f"drv_{drv}", use_container_width=True,
                        ):
                            st.session_state.nav_path = drv
                            st.rerun()

            # List subdirectories
            subdirs = _list_subdirs(nav)
            if subdirs:
                chosen = st.selectbox(
                    "Subdirectories",
                    subdirs,
                    label_visibility="collapsed",
                    key="nav_subdir_select",
                )
                if st.button(
                    f"📂 Open **{chosen}**",
                    key="nav_open",
                    use_container_width=True,
                ):
                    st.session_state.nav_path = os.path.join(nav, chosen)
                    st.rerun()
            else:
                st.caption("_(no subdirectories)_")

        st.divider()

        # ── Dynamic Evaluation Criteria ──
        st.markdown("## 📋 Evaluation Criteria")

        active_criteria: list[str] = []

        # Default criteria
        for i, criterion in enumerate(DEFAULT_CRITERIA):
            if st.checkbox(criterion, value=True, key=f"dc_{i}"):
                active_criteria.append(criterion)

        # Custom criteria
        if st.session_state.custom_criteria:
            st.caption("Custom Criteria")
            idx_to_remove: int | None = None
            for i, criterion in enumerate(st.session_state.custom_criteria):
                col_chk, col_btn = st.columns([6, 1])
                with col_chk:
                    if st.checkbox(criterion, value=True, key=f"cc_{i}"):
                        active_criteria.append(criterion)
                with col_btn:
                    if st.button("🗑️", key=f"rm_{i}", help="Remove this criterion"):
                        idx_to_remove = i
            if idx_to_remove is not None:
                st.session_state.custom_criteria.pop(idx_to_remove)
                st.rerun()

        # Add new custom criterion
        st.caption("Add Custom")
        col_inp, col_add = st.columns([4, 1])
        with col_inp:
            new_crit = st.text_input(
                "New criterion",
                label_visibility="collapsed",
                placeholder="e.g. Used proper commenting?",
            )
        with col_add:
            if st.button("➕", help="Add criterion") and new_crit and new_crit.strip():
                st.session_state.custom_criteria.append(new_crit.strip())
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # Main Area
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<h1 class="hero-title">📝 Automated Assignment Grader</h1>'
        '<p class="hero-sub">'
        "Two-stage Gemini pipeline for batch grading handwritten PDFs"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Active schema preview ──
    if active_criteria:
        with st.expander("📐 Active Evaluation Schema", expanded=False):
            schema_model = build_evaluation_schema(active_criteria)
            st.json(schema_model.model_json_schema(), expanded=2)

    # ── Input validation ──
    ready = True
    if not api_key:
        st.info("🔑 Enter your OpenRouter API key in the sidebar to begin.")
        ready = False
    if not question_file:
        st.info("📄 Upload the question PDF in the sidebar.")
        ready = False
    if not master_file:
        st.info("📄 Upload the master solution PDF in the sidebar.")
        ready = False
    if not target_dir or not os.path.isdir(target_dir):
        if target_dir:
            st.warning(f"📁 Directory not found: `{target_dir}`")
        else:
            st.info("📁 Specify a valid target directory containing student PDFs.")
        ready = False
    if not active_criteria:
        st.warning("📋 Select at least one evaluation criterion.")
        ready = False

    if not ready:
        return

    # ── Directory scan ──
    master_filename = master_file.name if master_file else None
    student_pdfs = discover_student_pdfs(target_dir, master_filename)

    if not student_pdfs:
        st.warning("No student PDF files found in the target directory.")
        return

    st.success(f"Found **{len(student_pdfs)}** student PDF(s) in `{target_dir}`")

    # ── Current DB status (if tracker already exists) ──
    db_path = os.path.join(target_dir, DB_NAME)
    if os.path.isfile(db_path):
        conn_peek = init_db(target_dir)
        counts = get_status_counts(conn_peek)
        if counts:
            status_icons = {
                "PENDING": "🟡",
                "MAPPED": "🔵",
                "EVALUATED": "🟢",
                "FAILED": "🔴",
            }
            cols = st.columns(len(counts))
            for col, (status, count) in zip(cols, sorted(counts.items())):
                icon = status_icons.get(status, "⚪")
                col.metric(f"{icon} {status}", count)

    st.divider()

    # ── Launch button ──
    if st.button(
        "🚀 Start Batch Processing",
        type="primary",
        use_container_width=True,
    ):
        _run_batch_pipeline(
            api_key=api_key,
            model_id=model_id,
            master_file=master_file,
            question_file=question_file,
            target_dir=target_dir,
            student_pdfs=student_pdfs,
            active_criteria=active_criteria,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Script Entry
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
