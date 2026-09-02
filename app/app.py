"""FaceTrace — Hackathon-Ready Streamlit Application (Phase 5).

End-to-end biometric provenance engine with genuine web search:
  Upload → Face Detection → SerpApi Google Lens Search → Candidate Ranking
  → Evidence Fingerprinting → Blockchain Anchoring → Verification & Tamper Detection
"""

import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

from src.core.config import get_settings
from src.core.types import PlatformType, VerificationStatus
from src.evidence.models import EvidenceRecord
from src.pipeline.base import PipelineOutput, PipelineStage
from src.pipeline.orchestrator import DefaultFaceTracePipeline
from src.verification.engine import verify_evidence

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Page Configuration
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceTrace | Biometric Provenance Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
# Custom CSS — Dark Professional Theme
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Root variables */
:root {
    --bg-primary: #0F172A;
    --bg-card: #1E293B;
    --bg-card-hover: #263549;
    --border-subtle: #334155;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --accent-blue: #3B82F6;
    --accent-purple: #8B5CF6;
    --accent-green: #10B981;
    --accent-red: #EF4444;
    --accent-amber: #F59E0B;
    --accent-pink: #EC4899;
    --accent-cyan: #06B6D4;
}

/* Global font */
html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main background */
.stApp {
    background: var(--bg-primary) !important;
}

/* Main title gradient */
.main-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.subtitle {
    color: var(--text-secondary);
    font-size: 1.0rem;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* Pipeline breadcrumb */
.pipeline-breadcrumb {
    display: flex;
    gap: 4px;
    align-items: center;
    flex-wrap: wrap;
    margin: 1rem 0;
    padding: 12px 16px;
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
}

.pipeline-step {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(59, 130, 246, 0.1);
    color: var(--text-secondary);
    transition: all 0.3s ease;
}

.pipeline-step.active {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
}

.pipeline-step.completed {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
}

.pipeline-step.failed {
    background: rgba(239, 68, 68, 0.15);
    color: var(--accent-red);
}

.pipeline-arrow {
    color: var(--text-muted);
    font-size: 0.7rem;
    margin: 0 2px;
}

/* Dark card */
.dark-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    border: 1px solid var(--border-subtle);
    transition: border-color 0.2s ease;
}

.dark-card:hover {
    border-color: var(--accent-blue);
}

.dark-card h4 {
    margin-top: 0;
    color: var(--text-primary);
    font-weight: 700;
    font-size: 1.05rem;
}

/* Score bar */
.score-bar-bg {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    overflow: hidden;
    margin: 6px 0;
}

.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.score-bar-fill.high { background: linear-gradient(90deg, #10B981, #34D399); }
.score-bar-fill.medium { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
.score-bar-fill.low { background: linear-gradient(90deg, #6B7280, #9CA3AF); }

/* Status badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-match { background: rgba(16, 185, 129, 0.15); color: #34D399; }
.badge-no-match { background: rgba(100, 116, 139, 0.15); color: #94A3B8; }
.badge-error { background: rgba(239, 68, 68, 0.15); color: #F87171; }
.badge-verified { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-tampered { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-not-found { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Platform badges */
.platform-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.platform-instagram { background: rgba(236, 72, 153, 0.15); color: #F472B6; }
.platform-facebook { background: rgba(59, 130, 246, 0.2); color: #93C5FD; }
.platform-twitter { background: rgba(59, 130, 246, 0.15); color: #60A5FA; }
.platform-linkedin { background: rgba(6, 182, 212, 0.15); color: #22D3EE; }
.platform-reddit { background: rgba(245, 158, 11, 0.15); color: #FBBF24; }
.platform-wikipedia { background: rgba(148, 163, 184, 0.15); color: #CBD5E1; }
.platform-youtube { background: rgba(239, 68, 68, 0.15); color: #F87171; }
.platform-pinterest { background: rgba(239, 68, 68, 0.15); color: #F87171; }
.platform-news { background: rgba(168, 85, 247, 0.15); color: #C084FC; }
.platform-web { background: rgba(148, 163, 184, 0.1); color: #94A3B8; }
.platform-unknown { background: rgba(148, 163, 184, 0.1); color: #94A3B8; }

/* Hash display */
.hash-display {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.78rem;
    color: var(--accent-cyan);
    word-break: break-all;
    background: rgba(6, 182, 212, 0.08);
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid rgba(6, 182, 212, 0.15);
}

/* Blockchain receipt */
.receipt-table {
    width: 100%;
    border-collapse: collapse;
}

.receipt-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.85rem;
}

.receipt-table td:first-child {
    color: var(--text-muted);
    font-weight: 600;
    width: 35%;
}

.receipt-table td:last-child {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

/* Verification banner */
.verification-banner {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 16px 0;
}

.verification-banner.verified {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(52, 211, 153, 0.08));
    color: #34D399;
    border: 2px solid rgba(16, 185, 129, 0.3);
}

.verification-banner.tampered {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(248, 113, 113, 0.08));
    color: #F87171;
    border: 2px solid rgba(239, 68, 68, 0.3);
}

.verification-banner.not-found {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(251, 191, 36, 0.08));
    color: #FBBF24;
    border: 2px solid rgba(245, 158, 11, 0.3);
}

.verification-banner.error {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(248, 113, 113, 0.08));
    color: #F87171;
    border: 2px solid rgba(239, 68, 68, 0.3);
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 12px 0;
}

.section-header .icon {
    font-size: 1.3rem;
}

.section-header .text {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
}

.section-header .desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-left: auto;
}

/* Metric card */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}

.metric-item {
    flex: 1;
    background: var(--bg-card);
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid var(--border-subtle);
    text-align: center;
}

.metric-item .value {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--accent-blue);
}

.metric-item .label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* Hide Streamlit default menu and footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0B1120 !important;
    border-right: 1px solid var(--border-subtle);
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Session State Initialization
# ────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state keys used by the application."""
    defaults = {
        "app_stage": "IDLE",
        "uploaded_image_bytes": None,
        "uploaded_filename": None,
        "pipeline_output": None,
        "pipeline_running": False,
        "tamper_test_result": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()
settings = get_settings()


# ────────────────────────────────────────────────────────────
# Helper Functions & Resource Cache
# ────────────────────────────────────────────────────────────
@st.cache_resource
def get_cached_pipeline_instance():
    """Cache pipeline instance and models across Streamlit re-renders."""
    return DefaultFaceTracePipeline()


def get_platform_class(platform: str) -> str:
    """Return CSS class for platform badge."""
    p = platform.lower()
    if p in ("instagram", "facebook", "twitter", "linkedin", "reddit", "wikipedia", "youtube", "pinterest", "news"):
        return f"platform-{p}"
    return "platform-web"


def get_score_class(score: float) -> str:
    """Return CSS class for score bar fill."""
    if score >= 0.45:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def get_status_badge(status: str) -> str:
    """Return badge HTML for match status."""
    s = status.upper()
    if s == "MATCH":
        return '<span class="badge badge-match">✓ MATCH</span>'
    if s == "NO_MATCH":
        return '<span class="badge badge-no-match">✗ NO MATCH</span>'
    if s == "NO_FACE_DETECTED":
        return '<span class="badge badge-no-match">⚠ NO FACE</span>'
    return f'<span class="badge badge-error">{s}</span>'


def render_pipeline_breadcrumb(current_stage: str):
    """Render the 7-step pipeline breadcrumb with active/completed states."""
    stages = [
        ("📁", "Upload"),
        ("👤", "Detect"),
        ("🌐", "Search"),
        ("🎯", "Match"),
        ("📜", "Evidence"),
        ("⛓️", "Blockchain"),
        ("✅", "Verify"),
    ]

    stage_map = {
        "IDLE": -1,
        "UPLOADED": 0,
        "FACE_DETECTION": 1, "FACE_ENCODING": 1,
        "SEARCHING": 2,
        "CANDIDATE_EVALUATION": 3, "MATCHING": 3,
        "EVIDENCE_EXTRACTION": 4, "FINGERPRINTING": 4,
        "BLOCKCHAIN_RECORDING": 5,
        "VERIFICATION": 6,
        "COMPLETED": 7,
        "FAILED": -2,
    }

    current_idx = stage_map.get(current_stage, -1)

    html = '<div class="pipeline-breadcrumb">'
    for i, (icon, label) in enumerate(stages):
        if current_stage == "FAILED":
            cls = "pipeline-step"
        elif i < current_idx:
            cls = "pipeline-step completed"
        elif i == current_idx:
            cls = "pipeline-step active"
        else:
            cls = "pipeline-step"

        html += f'<div class="{cls}">{icon} {label}</div>'
        if i < len(stages) - 1:
            html += '<span class="pipeline-arrow">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Sidebar: System Status & Configuration
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FaceTrace")
    st.caption("Biometric Provenance Engine v0.5.0 (Phase 5)")

    st.divider()

    st.markdown("### 🌐 Search Provider")
    is_serpapi = settings.SEARCH_PROVIDER.lower() == "serpapi"
    has_serp_key = bool(settings.SERPAPI_API_KEY)

    if is_serpapi and has_serp_key:
        st.markdown(
            '<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); '
            'padding: 10px; border-radius: 8px; margin-bottom: 8px;">'
            '<span style="color: #34D399; font-weight: 700; font-size: 0.85rem;">🌐 LIVE WEB SEARCH</span>'
            '<div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">SerpApi Google Lens engine active</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif is_serpapi and not has_serp_key:
        st.markdown(
            '<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); '
            'padding: 10px; border-radius: 8px; margin-bottom: 8px;">'
            '<span style="color: #F87171; font-weight: 700; font-size: 0.85rem;">⚠️ LIVE SEARCH (NO API KEY)</span>'
            '<div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">Set SERPAPI_API_KEY in .env or switch to mock</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.warning("SerpApi is configured but `SERPAPI_API_KEY` is missing in `.env`. Set your API key or set `SEARCH_PROVIDER=mock`.")
    else:
        st.markdown(
            '<div style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); '
            'padding: 10px; border-radius: 8px; margin-bottom: 8px;">'
            '<span style="color: #60A5FA; font-weight: 700; font-size: 0.85rem;">🧪 LOCAL DEMO</span>'
            '<div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">Offline mock provider active</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### ⚙️ Pipeline Configuration")
    st.markdown(f"**Face Detector:** `{settings.FACE_DETECTOR_PROVIDER}`")
    st.markdown(f"**Blockchain Provider:** `{settings.BLOCKCHAIN_PROVIDER}`")
    st.markdown(f"**Similarity Threshold:** `{settings.FACE_SIMILARITY_THRESHOLD}`")
    max_cands = getattr(settings, "MAX_CANDIDATES", getattr(settings, "MAX_SEARCH_RESULTS", 20))
    st.markdown(f"**Max Candidates:** `{max_cands}`")

    st.divider()

    st.markdown("### 🔒 Privacy Principles")
    st.info(
        "**1.** Biometric matching runs entirely **off-chain** and locally.\n\n"
        "**2.** Blockchain stores only a **32-byte SHA-256 hash** — never images or embeddings.\n\n"
        "**3.** Search queries only the query image for reverse discovery — embeddings are never sent to search providers.\n\n"
        "**4.** Search is strictly limited to publicly accessible and indexed content."
    )

    st.divider()

    st.markdown("### ℹ️ Public Search Scope")
    st.caption(
        "Results are limited to publicly accessible/indexed web content returned by the search provider. "
        "Face similarity is probabilistic; scores indicate mathematical similarity, not absolute proof of identity."
    )


# ────────────────────────────────────────────────────────────
# Main Header
# ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">FaceTrace Biometric Provenance Engine</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Face Detection → Genuine Web Search → Biometric Similarity Ranking → '
    'Cryptographic Evidence Fingerprint → Blockchain Anchoring → Integrity Verification'
    '</div>',
    unsafe_allow_html=True,
)

# Breadcrumb — show pipeline progress
pipeline_output = st.session_state.get("pipeline_output")
if pipeline_output:
    render_pipeline_breadcrumb(pipeline_output.stage.value)
else:
    render_pipeline_breadcrumb(st.session_state.get("app_stage", "IDLE"))


# ────────────────────────────────────────────────────────────
# Section 1: Image Upload
# ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">'
    '<span class="icon">📁</span>'
    '<span class="text">Upload Query Face</span>'
    '<span class="desc">JPEG or PNG • Single face • Consented portrait image</span>'
    '</div>',
    unsafe_allow_html=True,
)

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a clear portrait image",
        type=["jpg", "jpeg", "png"],
        help="Use a consented demo image or public figure photo. Must contain exactly one face.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.session_state["uploaded_image_bytes"] = image_bytes
        st.session_state["uploaded_filename"] = uploaded_file.name
        if st.session_state["app_stage"] == "IDLE":
            st.session_state["app_stage"] = "UPLOADED"

with col_preview:
    if st.session_state["uploaded_image_bytes"]:
        img = Image.open(io.BytesIO(st.session_state["uploaded_image_bytes"]))
        st.image(
            img,
            caption=f"📷 {st.session_state['uploaded_filename']} ({img.width}×{img.height})",
            use_container_width=True,
        )


# ────────────────────────────────────────────────────────────
# Section 2: Pipeline Execution
# ────────────────────────────────────────────────────────────
if st.session_state["uploaded_image_bytes"] and st.session_state["app_stage"] in ("UPLOADED", "RESULTS", "ERROR"):
    st.markdown(
        '<div class="section-header">'
        '<span class="icon">🚀</span>'
        '<span class="text">Run Search & Biometric Matching</span>'
        '<span class="desc">Discover candidates & evaluate biometric similarity</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        auto_anchor = st.checkbox("Automatically anchor verified match to blockchain upon search", value=False)
    with col_btn2:
        btn_label = "🔍 Run Search & Matching"

    if st.button(btn_label, type="primary", use_container_width=True):
        st.session_state["app_stage"] = "PROCESSING"
        st.session_state["pipeline_output"] = None
        st.session_state["tamper_test_result"] = None

        with st.status("🔬 Executing FaceTrace Pipeline...", expanded=True) as status:
            try:
                progress_log = []

                def on_progress(stage: str, message: str):
                    progress_log.append((stage, message))
                    st.write(f"✓ **{stage}** — {message}")

                pipeline = DefaultFaceTracePipeline(
                    progress_callback=on_progress,
                )

                start_time = time.time()
                # Run search and matching (or full auto_anchor if checked)
                output = pipeline.run(
                    st.session_state["uploaded_image_bytes"],
                    auto_anchor=auto_anchor,
                )
                elapsed = time.time() - start_time

                st.session_state["pipeline_output"] = output

                if output.stage == PipelineStage.FAILED:
                    st.session_state["app_stage"] = "ERROR"
                    status.update(label=f"❌ Pipeline failed — {output.error_message}", state="error")
                else:
                    st.session_state["app_stage"] = "RESULTS"
                    match_count = sum(1 for m in output.all_matches if m.is_match)
                    status.update(
                        label=f"✅ Search & evaluation completed in {elapsed:.1f}s — {match_count} match(es) discovered",
                        state="complete",
                    )

            except Exception as exc:
                st.session_state["app_stage"] = "ERROR"
                status.update(label=f"❌ Critical error: {exc}", state="error")
                st.error(f"Pipeline crashed: {exc}")

        st.rerun()


# ────────────────────────────────────────────────────────────
# Section 3: Results Display
# ────────────────────────────────────────────────────────────
output: PipelineOutput | None = st.session_state.get("pipeline_output")

if output is not None:
    # ── Error State ──
    if output.stage == PipelineStage.FAILED:
        st.markdown(
            '<div class="section-header">'
            '<span class="icon">❌</span>'
            '<span class="text">Pipeline Error</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.error(f"**Error:** {output.error_message}")
        if "SerpApi" in str(output.error_message) and not settings.SERPAPI_API_KEY:
            st.info("💡 **Tip:** Set `SERPAPI_API_KEY` in `.env` for real Google Lens searches, or switch to `SEARCH_PROVIDER=mock` in `.env` to run the offline demonstration.")

    # ── Face Detection Info ──
    if output.query_face:
        st.markdown(
            '<div class="section-header">'
            '<span class="icon">👤</span>'
            '<span class="text">Query Face Validation</span>'
            '<span class="desc">InsightFace SCRFD + ArcFace 512D</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            bbox = output.query_face.bbox
            st.metric("Bounding Box", f"[{bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}]")
        with c2:
            st.metric("Detection Confidence", f"{output.query_face.confidence:.3f}")
        with c3:
            dim = output.query_embedding.dimension if output.query_embedding else "N/A"
            st.metric("Embedding Dimension", f"{dim}D")

    # ── Discovered Candidates & Match Ranking ──
    if output.all_matches:
        st.markdown(
            '<div class="section-header">'
            '<span class="icon">🎯</span>'
            '<span class="text">Discovered Candidates & Facial Matches</span>'
            f'<span class="desc">{len(output.all_matches)} candidates evaluated</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Summary metrics
        total = len(output.all_matches)
        matched = sum(1 for m in output.all_matches if m.is_match)
        no_match = sum(1 for m in output.all_matches if m.status == "NO_MATCH")
        no_face = sum(1 for m in output.all_matches if m.status == "NO_FACE_DETECTED")
        errors = sum(1 for m in output.all_matches if m.status == "ERROR")

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item">
                <div class="value" style="color: #3B82F6;">{total}</div>
                <div class="label">Total Candidates Discovered</div>
            </div>
            <div class="metric-item">
                <div class="value" style="color: #10B981;">{matched}</div>
                <div class="label">Biometric Matches</div>
            </div>
            <div class="metric-item">
                <div class="value" style="color: #94A3B8;">{no_match}</div>
                <div class="label">Below Threshold</div>
            </div>
            <div class="metric-item">
                <div class="value" style="color: #F59E0B;">{no_face}</div>
                <div class="label">No Face Detected</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Individual candidate cards
        for i, match in enumerate(output.all_matches):
            platform_cls = get_platform_class(match.candidate.platform.value)
            score_cls = get_score_class(match.similarity_score)
            score_pct = max(0, min(100, match.similarity_score * 100))

            with st.container():
                cols = st.columns([3, 1, 1])

                with cols[0]:
                    st.markdown(
                        f'<span class="platform-badge {platform_cls}">{match.candidate.platform.value}</span> '
                        f'&nbsp; **{match.candidate.title[:75]}**',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="score-bar-bg">'
                        f'<div class="score-bar-fill {score_cls}" style="width: {score_pct}%"></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"🌐 **Discovery URL:** {match.candidate.url[:85]}")

                with cols[1]:
                    st.markdown(f"**Similarity:** `{match.similarity_score:.4f}` ({score_pct:.1f}%)")
                    st.caption(f"Faces found: {match.detected_faces_count} | Threshold: {match.threshold_used:.2f}")

                with cols[2]:
                    st.markdown(get_status_badge(match.status), unsafe_allow_html=True)

                # Show candidate image if available (bytes, local path, or URL)
                has_image = bool(match.candidate.image_bytes or match.local_image_path or match.candidate.image_url)
                if has_image:
                    with st.expander(f"🖼️ View candidate image — {match.candidate.title[:45]}"):
                        if match.candidate.image_bytes:
                            st.image(io.BytesIO(match.candidate.image_bytes), use_container_width=True)
                        elif match.local_image_path and Path(match.local_image_path).exists():
                            st.image(match.local_image_path, use_container_width=True)
                        elif match.candidate.image_url and match.candidate.image_url.startswith("http"):
                            st.image(match.candidate.image_url, use_container_width=True)

                if i < len(output.all_matches) - 1:
                    st.markdown("---")

    # ── Section 4: Match Experience & User-Directed Anchoring ──
    if output.best_match is not None:
        best_match = output.best_match
        platform_cls = get_platform_class(best_match.candidate.platform.value)

        st.markdown(
            '<div class="section-header">'
            '<span class="icon">🏆</span>'
            '<span class="text">Verified Match Experience</span>'
            '<span class="desc">Highest confidence biometric match</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"""
        <div class="dark-card" style="border-left: 4px solid var(--accent-green); background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(30, 41, 59, 0.95));">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="badge badge-match" style="font-size: 0.85rem; padding: 6px 16px;">✓ MATCH FOUND</span>
                <span style="color: #34D399; font-weight: 700; font-size: 1.1rem;">Biometric Similarity: {best_match.similarity_score:.2%}</span>
            </div>
            <table class="receipt-table">
                <tr><td>Platform</td><td><span class="platform-badge {platform_cls}">{best_match.candidate.platform.value}</span></td></tr>
                <tr><td>Source URL</td><td><a href="{best_match.candidate.url}" target="_blank" style="color: #60A5FA;">{best_match.candidate.url}</a></td></tr>
                <tr><td>Face Similarity</td><td><b>{best_match.similarity_score:.4f} ({best_match.similarity_score:.2%})</b></td></tr>
                <tr><td>Similarity Threshold</td><td>{best_match.threshold_used:.4f} ({best_match.threshold_used:.2%})</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # User choice: Anchor evidence to blockchain
        if not output.evidence_record or not output.anchor_result:
            st.markdown("#### ⚓ Anchor Evidence to Blockchain")
            st.caption("Submit the cryptographic evidence fingerprint to the EvidenceRegistry smart contract to establish an immutable, tamper-evident audit trail.")

            if st.button("⛓️ Anchor Verified Evidence to Blockchain", type="primary", use_container_width=True):
                with st.spinner("Packaging canonical evidence & submitting on-chain transaction..."):
                    try:
                        pipeline = DefaultFaceTracePipeline()
                        output = pipeline.anchor_match(output, best_match)
                        st.session_state["pipeline_output"] = output
                        st.success("Evidence successfully anchored and verified on-chain!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Blockchain anchoring failed: {exc}")

    # ── Evidence Record ──
    if output.evidence_record:
        st.markdown(
            '<div class="section-header">'
            '<span class="icon">📜</span>'
            '<span class="text">Evidence Record</span>'
            '<span class="desc">Cryptographic provenance metadata</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        er = output.evidence_record
        col_ev1, col_ev2 = st.columns(2)

        with col_ev1:
            st.markdown(f"""
            <div class="dark-card">
                <h4>📋 Canonical Evidence Fields</h4>
                <table class="receipt-table">
                    <tr><td>Post URL</td><td>{er.post_url}</td></tr>
                    <tr><td>Platform</td><td>{er.platform}</td></tr>
                    <tr><td>Title</td><td>{er.title[:60]}</td></tr>
                    <tr><td>Similarity Score</td><td>{er.similarity_score:.4f}</td></tr>
                    <tr><td>Threshold</td><td>{er.threshold:.4f}</td></tr>
                    <tr><td>Discovered At</td><td>{er.discovered_at}</td></tr>
                    <tr><td>Version</td><td>{er.evidence_version}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_ev2:
            st.markdown(f"""
            <div class="dark-card">
                <h4>🔐 Cryptographic Hashes</h4>
                <p style="color: var(--text-muted); font-size: 0.8rem; margin-bottom: 8px;">IMAGE SHA-256 (raw candidate image bytes)</p>
                <div class="hash-display">{er.image_sha256}</div>
                <br>
                <p style="color: var(--text-muted); font-size: 0.8rem; margin-bottom: 8px;">EVIDENCE SHA-256 (canonical JSON hash)</p>
                <div class="hash-display">{output.fingerprint}</div>
                <br>
                <p style="color: var(--text-muted); font-size: 0.8rem; margin-bottom: 8px;">EVM bytes32 Fingerprint</p>
                <div class="hash-display">{er.to_bytes32()}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📄 View Canonical JSON Representation"):
            st.code(er.to_canonical_json(), language="json")

    # ── Blockchain Anchoring Result ──
    if output.anchor_result:
        st.markdown(
            '<div class="section-header">'
            '<span class="icon">⛓️</span>'
            '<span class="text">Blockchain Anchoring</span>'
            f'<span class="desc">Network: {settings.BLOCKCHAIN_PROVIDER}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        ar = output.anchor_result

        if ar.is_success:
            ts_str = (
                datetime.fromtimestamp(ar.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                if ar.timestamp else "N/A"
            )

            st.markdown(f"""
            <div class="dark-card">
                <h4>✅ Transaction Confirmed</h4>
                <table class="receipt-table">
                    <tr><td>Transaction Hash</td><td>{ar.tx_hash}</td></tr>
                    <tr><td>Block Number</td><td>{ar.block_number}</td></tr>
                    <tr><td>Contract Address</td><td>{ar.contract_address}</td></tr>
                    <tr><td>Status</td><td>{'✅ Confirmed' if ar.status == 1 else '❌ Failed'}</td></tr>
                    <tr><td>On-Chain Stored Hash</td><td>{ar.stored_evidence_hash}</td></tr>
                    <tr><td>Block Timestamp</td><td>{ts_str}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"**Anchoring Failed:** {ar.error}")

    # ── Verification Result ──
    if output.verification_result:
        st.markdown(
            '<div class="section-header">'
            '<span class="icon">✅</span>'
            '<span class="text">Integrity Verification</span>'
            '<span class="desc">On-chain evidence cryptographic integrity check</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        vr = output.verification_result
        status_str = vr.status.value

        if vr.is_verified:
            banner_cls = "verified"
            banner_icon = "✅"
            banner_text = "EVIDENCE VERIFIED — Integrity confirmed on blockchain"
        elif vr.status == VerificationStatus.TAMPERED:
            banner_cls = "tampered"
            banner_icon = "🚨"
            banner_text = "TAMPER DETECTED — Evidence has been modified"
        elif vr.status == VerificationStatus.NOT_FOUND:
            banner_cls = "not-found"
            banner_icon = "⚠️"
            banner_text = "NOT FOUND — Fingerprint not anchored on blockchain"
        else:
            banner_cls = "error"
            banner_icon = "❌"
            banner_text = f"ERROR — {status_str}"

        st.markdown(
            f'<div class="verification-banner {banner_cls}">{banner_icon} {banner_text}</div>',
            unsafe_allow_html=True,
        )

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(f"""
            <div class="dark-card">
                <h4>🔎 Verification Details</h4>
                <table class="receipt-table">
                    <tr><td>Status</td><td><span class="badge badge-{banner_cls}">{status_str}</span></td></tr>
                    <tr><td>Computed Fingerprint</td><td style="font-size:0.7rem;">{vr.computed_fingerprint}</td></tr>
                    <tr><td>Source URL</td><td>{vr.source_url}</td></tr>
                    <tr><td>Evidence Version</td><td>{vr.evidence_version}</td></tr>
                    <tr><td>Verified At</td><td>{vr.verification_timestamp}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_v2:
            if vr.on_chain_record:
                ocr = vr.on_chain_record
                st.markdown(f"""
                <div class="dark-card">
                    <h4>📦 On-Chain Record</h4>
                    <table class="receipt-table">
                        <tr><td>Record Exists</td><td>{'✅ Yes' if ocr.exists else '❌ No'}</td></tr>
                        <tr><td>On-Chain Fingerprint</td><td style="font-size:0.7rem;">{ocr.fingerprint}</td></tr>
                        <tr><td>Recorder Address</td><td style="font-size:0.7rem;">{ocr.recorder}</td></tr>
                        <tr><td>Block Number</td><td>{ocr.block_number}</td></tr>
                        <tr><td>Recorded Timestamp</td><td>{ocr.formatted_time}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="dark-card">
                    <h4>📦 On-Chain Record</h4>
                    <p style="color: var(--text-muted);">No on-chain record found for this fingerprint.</p>
                </div>
                """, unsafe_allow_html=True)

        # Interactive Tamper Detection Demonstration
        st.markdown("#### 🧪 Interactive Tamper Detection Demonstration")
        st.caption("Demonstrate cryptographic tampering protection: Simulate modifying the evidence metadata and verify against the on-chain record.")

        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            if st.button("🚨 Simulate Tampered Evidence Check", use_container_width=True):
                # Create a modified evidence record
                tampered_er = EvidenceRecord(
                    post_url=output.evidence_record.post_url,
                    platform=output.evidence_record.platform,
                    title="MODIFIED: Unauthorized Title Edit",
                    similarity_score=0.9999,  # Altered similarity
                    threshold=output.evidence_record.threshold,
                    image_sha256=output.evidence_record.image_sha256,
                    discovered_at=output.evidence_record.discovered_at,
                    evidence_version=output.evidence_record.evidence_version,
                )
                pipeline = DefaultFaceTracePipeline()
                tamper_vr = verify_evidence(tampered_er, pipeline._blockchain)
                st.session_state["tamper_test_result"] = tamper_vr

        if st.session_state.get("tamper_test_result"):
            t_vr = st.session_state["tamper_test_result"]
            st.markdown(
                '<div class="verification-banner tampered">🚨 TAMPER DETECTED — The modified evidence produces a different hash that does not match the on-chain record!</div>',
                unsafe_allow_html=True,
            )
            st.info(f"**Tamper explanation:** Original on-chain hash is anchored permanently. The altered payload produced `{t_vr.computed_fingerprint}`, which is not registered on the smart contract.")

    # ── Section 5: No Match Found Completed State ──
    if (output.stage == PipelineStage.COMPLETED
            and output.best_match is None
            and not output.evidence_record):
        st.markdown(
            '<div class="section-header">'
            '<span class="icon">🔎</span>'
            '<span class="text">Search Completed — No Match Found</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "**Search completed, but no sufficiently similar face was found.**\n\n"
            f"The reverse-image search executed successfully and discovered {len(output.search_results or [])} candidates, "
            f"but none exceeded the biometric similarity threshold (threshold ≥ {settings.FACE_SIMILARITY_THRESHOLD:.2f}).\n\n"
            "ℹ️ *Note: This indicates no matching face was discovered among publicly indexed results returned by the search provider. "
            "It does not guarantee that no copies exist online or in private accounts.*"
        )


# ────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center; color: var(--text-muted); font-size: 0.75rem; padding: 8px 0;">'
    'FaceTrace v0.5.0 — Biometric Provenance Engine — HH Goa 2026'
    '</div>',
    unsafe_allow_html=True,
)
