"""FaceTrace - Streamlit Application Entrypoint (Phase 1 Foundation & Architecture Demo)."""

import streamlit as st

from src.core.config import get_settings
from src.core.types import BlockchainNetwork, PlatformType, VerificationStatus

# Configure Page
st.set_page_config(
    page_title="FaceTrace | Biometric Provenance & Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stage-card {
        background: #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #3B82F6;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #10B981;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()

# Sidebar: System Status & Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/biometrics.png", width=64)
    st.title("FaceTrace Status")
    st.caption("Phase 1: Architecture & Project Foundation")

    st.divider()
    st.subheader("Configured Providers")
    st.write(f"**Face Detector:** `{settings.FACE_DETECTOR_PROVIDER}`")
    st.write(f"**Search Engine:** `{settings.SEARCH_PROVIDER}`")
    st.write(f"**Blockchain:** `{settings.BLOCKCHAIN_PROVIDER}`")
    st.write(f"**Match Threshold:** `{settings.FACE_SIMILARITY_THRESHOLD}`")

    st.divider()
    st.markdown("### Core Principles")
    st.info(
        "**Principle 1:** AI performs face similarity matching off-chain.\n\n"
        "**Principle 2:** Blockchain anchors cryptographic fingerprint (SHA-256) of canonical evidence.\n\n"
        "**Principle 3:** Genuine dynamic search across public indexed sources only."
    )

# Main Application Header
st.markdown('<div class="main-title">FaceTrace Biometric Provenance Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real Face Detection → Genuine Web/Social Search → Cryptographic Canonical Fingerprint → Sepolia Blockchain Registry → Independent Integrity Verification</div>',
    unsafe_allow_html=True,
)

# Pipeline Flow Diagram
st.subheader("1. Pipeline Workflow Architecture")

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.markdown("📁 **1. Upload**\n\nConsented face image")
with col2:
    st.markdown("👤 **2. Analysis**\n\nSCRFD + ArcFace 512D")
with col3:
    st.markdown("🌐 **3. Search**\n\nGoogle Lens / Bing")
with col4:
    st.markdown("🎯 **4. Match**\n\nCandidate cosine sim")
with col5:
    st.markdown("📜 **5. Evidence**\n\nCanonical JSON")
with col6:
    st.markdown("⛓️ **6. Blockchain**\n\nSepolia bytes32")
with col7:
    st.markdown("✅ **7. Verify**\n\nOn-chain re-check")

st.divider()

# Upload & Interactive Execution View (Phase 1 Ready Shell)
st.subheader("2. Live Execution Sandbox (Phase 1 Ready)")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### Input Query Image")
    uploaded_file = st.file_uploader(
        "Upload a clear portrait or public test image (JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Use consented demo image or public figure photo for testing.",
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Query Face Image", use_column_width=True)
        st.success("Image uploaded. Ready for Phase 2 pipeline execution.")

with col_right:
    st.markdown("#### Module Readiness Matrix")
    modules = [
        ("src/face", "InsightFace SCRFD + ArcFace 512D Embeddings", "Ready"),
        ("src/search", "SearchProvider Abstraction (SerpApi / Bing / Mock)", "Ready"),
        ("src/candidates", "Candidate Downloader & Cosine Similarity Ranking", "Ready"),
        ("src/evidence", "Deterministic Canonicalization & SHA-256 Hash", "Ready"),
        ("src/blockchain", "EvidenceRegistry Smart Contract & Sepolia Web3", "Ready"),
        ("src/verification", "On-Chain Tamper-Evident Verification Engine", "Ready"),
        ("src/pipeline", "End-to-end Pipeline Orchestrator", "Ready"),
    ]
    for module_path, desc, status in modules:
        with st.container():
            st.markdown(
                f"**`{module_path}`**: {desc} &nbsp; "
                f"<span class='status-badge'>{status}</span>",
                unsafe_allow_html=True,
            )

st.divider()
st.caption("FaceTrace v0.1.0 — HH Goa 2026 Shortlisting Task 3 Foundation.")
