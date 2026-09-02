# FaceTrace 🔍⛓️

> **Biometric Provenance, Reverse-Search Matching, and Tamper-Evident Blockchain Evidence Verification**  
> *Built for HH Goa 2026 Shortlisting Task 3*

---

## 1. Executive Summary & Problem Statement

FaceTrace is an end-to-end evidence discovery and provenance verification pipeline. When a user supplies a facial image (query), FaceTrace:
1. Detects and extracts normalized biometric feature embeddings using deep neural networks (InsightFace SCRFD + ArcFace).
2. Executes genuine, live web and social reverse-image queries across public search engines (Google Lens via SerpApi or Bing Visual Search) to discover real-world indexed pages containing matching faces.
3. Downloads candidate media, validates imagery, extracts candidate facial features, and computes mathematical cosine similarity against the query.
4. Synthesizes a deterministic **Evidence Object** encapsulating the candidate URL, platform, image SHA-256 hash, similarity score, and discovery timestamp.
5. Canonicalizes the evidence dictionary into a deterministic, RFC-compliant format and computes a 32-byte **SHA-256 cryptographic fingerprint**.
6. Anchors this fingerprint into an immutable Ethereum smart contract (`EvidenceRegistry.sol`) deployed on the **Sepolia Testnet**.
7. Independently re-verifies evidence integrity by recalculating the local hash and comparing it against the on-chain registry state.

---

## 2. Core Architectural Principles

* **AI Performs Biometric Matching; Blockchain Verifies Integrity**: Face similarity is probabilistic AI; blockchain provides cryptographic, tamper-evident existence and integrity guarantees. Blockchain does **not** "prove identity"—it proves that this exact evidence bundle existed at a verified timestamp and has not been altered.
* **Zero Private Data on Chain**: Raw images, biometric embeddings, and personal identifiers are **never** stored on the public blockchain. Only the 32-byte deterministic SHA-256 fingerprint (`bytes32`) is recorded.
* **Legitimate Public Discovery Only**: No scraping behind login gates, no CAPTCHA bypassing, no terms-of-service violations. FaceTrace queries legitimate public visual search engine APIs that index publicly accessible web and social pages.
* **Deterministic Provenance**: Canonical JSON serialization ensures that any party re-running the verification on the same evidence bundle obtains the identical SHA-256 hash down to the bit.

---

## 3. End-to-End Pipeline Architecture

```
[ INPUT FACE IMAGE ]
        │
        ▼
[ 1. FACE DETECTION & ALIGNMENT ] ──► InsightFace SCRFD (Face Bbox, 5 Landmarks)
        │
        ▼
[ 2. BIOMETRIC EMBEDDING ] ──────────► InsightFace ArcFace (512D Normalized Vector)
        │
        ▼
[ 3. GENUINE WEB/SOCIAL SEARCH ] ───► SearchProvider (SerpApi Google Lens / Bing Visual)
        │
        ▼
[ 4. CANDIDATE INGESTION ] ──────────► HTTP Fetcher (Content validation, Image SHA-256)
        │
        ▼
[ 5. CANDIDATE FACE ANALYSIS ] ──────► Candidate Face Detection & Embedding
        │
        ▼
[ 6. SIMILARITY SCORING ] ───────────► Cosine Similarity: dot(u, v) / (|u|*|v|)
        │
        ▼
[ 7. BEST MATCH SELECTION ] ─────────► Threshold Filtering (Score >= 0.45)
        │
        ▼
[ 8. EVIDENCE EXTRACTION ] ──────────► Deterministic Evidence Object
        │
        ▼
[ 9. CANONICALIZATION ] ─────────────► Sorted Keys, Unicode NFC, Fixed Float Precision
        │
        ▼
[ 10. CRYPTOGRAPHIC FINGERPRINT ] ───► SHA-256 Hash (32-byte hex / bytes32)
        │
        ▼
[ 11. BLOCKCHAIN ANCHORING ] ────────► Sepolia EvidenceRegistry.recordEvidence()
        │
        ▼
[ 12. INTEGRITY VERIFICATION ] ──────► On-Chain Re-verification & Tamper Detection
```

---

## 4. Module Decomposition

The project is structured under `src/` as loosely-coupled, high-cohesion modules:

```
FaceTrace/
├── contracts/
│   └── EvidenceRegistry.sol      # Solidity smart contract for Sepolia testnet
├── src/
│   ├── core/                     # Configuration, types, exceptions, logging
│   │   ├── config.py             # Pydantic Settings with env validation & secret masking
│   │   ├── exceptions.py         # Domain error hierarchy
│   │   ├── logging.py            # Structured logging with secret redactor
│   │   └── types.py              # Domain enums (PlatformType, VerificationStatus)
│   ├── face/                     # Biometric face detection & feature extraction
│   │   └── base.py               # BoundingBox, DetectedFace, FaceEmbedding, Cosine Sim
│   ├── search/                   # Genuine web and social reverse-image search
│   │   └── base.py               # SearchProvider abstraction and SearchResult model
│   ├── candidates/               # Candidate fetching, matching, and ranking
│   │   └── base.py               # Candidate, CandidateMatch, CandidateProcessor
│   ├── evidence/                 # Evidence models and deterministic canonicalization
│   │   ├── canonical.py          # RFC canonical JSON serialization & SHA-256
│   │   └── models.py             # Evidence dataclass, to_bytes32()
│   ├── blockchain/               # Smart contract ABI and Web3 provider
│   │   ├── abi.py                # EvidenceRegistry ABI definition
│   │   └── base.py               # BlockchainProvider, OnChainRecord, TransactionReceipt
│   ├── verification/             # Independent integrity verification
│   │   └── base.py               # VerificationEngine, VerificationResult
│   └── pipeline/                 # Pipeline orchestrator
│       └── base.py               # PipelineStage, PipelineOutput, FaceTracePipeline
├── app/
│   └── app.py                    # Streamlit interactive UI application
├── tests/                        # Comprehensive unit test suite
│   ├── conftest.py               # Reusable fixtures
│   ├── test_config.py            # Config validation and secret masking tests
│   ├── test_evidence_canonicalization.py # Deterministic hashing & tamper tests
│   ├── test_face_interface.py    # Face models & cosine similarity tests
│   ├── test_search_interface.py  # Search models & platform inference tests
│   └── test_blockchain_interface.py # Contract ABI & verification status tests
├── scripts/
│   └── check_env.py              # System environment & dependency diagnostics
├── data/                         # Local cache and consented demo images
├── .env.example                  # Template configuration (NO SECRETS)
├── .gitignore                    # Git exclusions
├── requirements.txt              # Pinned dependencies
├── pyproject.toml                # Packaging & tool configurations
└── README.md                     # Complete project documentation
```

---

## 5. Technology Stack & Justification

| Layer | Technology | Justification |
| :--- | :--- | :--- |
| **Face Detection & Embedding** | InsightFace (SCRFD + ArcFace) with ONNX Runtime | SOTA accuracy (99.8% LFW), normalized 512D embeddings, pure ONNX execution avoiding C++ `dlib` compilation hurdles on Windows. |
| **Search Provider** | SerpApi (Google Lens engine) / Bing Visual Search | Direct social media APIs (Instagram/X) prohibit arbitrary image queries. Google Lens and Bing index public social posts legitimately, accepting image uploads. |
| **Evidence Canonicalization** | Python `hashlib` + RFC 8785 JSON rules | Guarantees bitwise-identical SHA-256 fingerprints across platforms, key orderings, and machines. |
| **Blockchain Testnet** | Ethereum Sepolia (`web3.py`, `eth-account`) | Robust, widely supported EVM testnet. Smart contract stores 32-byte `bytes32` hashes with zero gas for read queries. |
| **User Interface** | Streamlit | Clean, interactive local dashboard demonstrating each sequential pipeline stage. |
| **Config & Safety** | Pydantic Settings | Type-safe environment loading with automatic secret masking to prevent credentials leakage. |

---

## 6. Setup & Installation

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git

### Step 1: Clone and Create Virtual Environment
```bash
git clone <repository-url>
cd FaceTrace

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
*(Never commit `.env` to Git. Private keys and API keys are strictly protected).*

### Step 4: Run Diagnostic Check
```bash
python scripts/check_env.py
```

### Step 5: Run Unit Tests
```bash
pytest -v
```

### Step 6: Deploy Local EvidenceRegistry Contract
```bash
python scripts/deploy_contract.py
```

### Step 7: Run End-to-End Local Pipeline Demo
```bash
python scripts/demo_phase3.py
```

### Step 8: Launch Streamlit Demo App
```bash
streamlit run app/app.py
```

---

## 7. Phase 3: Evidence Fingerprinting & Blockchain Anchoring

### EvidenceRecord Structure
The `EvidenceRecord` represents the verified match and encapsulates non-biometric provenance metadata:
```json
{
  "post_url": "https://instagram.com/p/C_abc123xyz/",
  "platform": "instagram",
  "title": "Jane Doe (@janedoe) • Instagram portrait photo (Match 01)",
  "image_sha256": "3fd9c94c5ea2ba4cbf8aa531502f7f1da450901f3f45225ab84fd3b6c95399a6",
  "similarity_score": 1.0,
  "threshold": 0.45,
  "discovered_at": "2026-09-01T18:45:28Z",
  "evidence_version": "1.0.0",
  "caption": "Verified public matching social portrait."
}
```

### Image Hash vs. Evidence Hash
FaceTrace enforces a strict conceptual and cryptographic separation between:
1. **IMAGE HASH (`image_sha256`)**:
   - The SHA-256 cryptographic digest of the raw candidate image bytes.
   - Proves exact bitwise integrity of the media file itself.
2. **EVIDENCE HASH (`evidence_sha256`)**:
   - The SHA-256 cryptographic digest of the canonical RFC 8785 JSON representation of the entire `EvidenceRecord`.
   - Binds the candidate image hash, source URL, platform, similarity score, threshold, and discovery timestamp together into an unforgeable 32-byte fingerprint.

### Canonicalization Process (RFC 8785)
To ensure identical evidence bundles produce bitwise-identical SHA-256 digests across different programming languages, platforms, and dictionary key orderings:
1. Dictionary keys are sorted in strict lexicographical order.
2. Compact separators without extraneous whitespace (`","`, `":"`) are used.
3. Strings are normalized using Unicode Normalization Form C (NFC).
4. Floating-point numbers (similarity score, threshold) are rounded to 4 decimal places to prevent float precision drift.
5. UTF-8 serialization without ASCII escaping ensures cross-platform consistency.

### Smart Contract: `EvidenceRegistry.sol`
* **Purpose**: Provides an immutable, append-only registry recording cryptographic evidence fingerprints and block timestamps.
* **Network**: Compatible with local EVM (via `web3.EthereumTesterProvider` or `http://127.0.0.1:8545`) and Ethereum Sepolia Testnet.
* **On-Chain Storage (`_records[bytes32]`)**:
  - `bytes32 fingerprint`: 32-byte SHA-256 hash of the canonical evidence bundle.
  - `address recorder`: EVM account address that submitted the anchoring transaction.
  - `uint256 timestamp`: Block timestamp when the transaction was mined.
  - `uint256 blockNumber`: Block number in which the transaction was included.
  - `string sourceUrl`: Public web/social URL where evidence was legitimately discovered.
* **What is NOT Stored On-Chain (Strict Privacy Guarantee)**:
  - NO raw face scans or selfies.
  - NO biometric embeddings or 512D ArcFace feature vectors.
  - NO image binary data.
  - NO personal or private user identifiers.

### Blockchain Deployment & Verification
1. **Deployment**: Run `python scripts/deploy_contract.py` to compile `EvidenceRegistry.sol` using `solc 0.8.20` and deploy to the active network.
2. **Anchoring**: `EthereumBlockchainService.anchor_evidence(bytes32, sourceUrl)` submits a transaction calling `recordEvidence`, awaits receipt, and verifies confirmed on-chain storage.
3. **Verification**: `verify_evidence(evidence_record)` recalculates the canonical fingerprint and performs a zero-gas `getEvidence(bytes32)` view call to confirm existence and metadata consistency.
4. **Tamper Detection**: If any field in the evidence record is altered (e.g. similarity score `1.0000` -> `0.8300`), the recalculated fingerprint changes completely (SHA-256 avalanche effect) and results in `TAMPERED / INVALID`.

> 📖 **Full Phase 3 Guide**: See [`docs/phase3_evidence_blockchain.md`](docs/phase3_evidence_blockchain.md) for a detailed judge-friendly breakdown of the evidence model, hashing mechanics, canonicalization, and zero-biometrics privacy guarantees.

### Running Phase 3 Demo & Tests
```bash
# Run the 24-point Phase 3 test suite
python -m pytest tests/test_phase3_blockchain.py -v

# Run the complete test suite (74 tests)
python -m pytest -v

# Run the end-to-end local blockchain demonstration
python scripts/demo_phase3.py
```

---

## 7. Security, Privacy & Ethics

1. **Consent First**: Only consented demo images or public figures are used for demonstration.
2. **Zero Biometrics on Chain**: Raw biometrics never touch public ledgers.
3. **No Credential Exposure**: Logging redacts private keys and tokens. `.gitignore` shields secrets.
4. **Legitimate Access**: Respects robots.txt, public index boundaries, and does not bypass access control mechanisms.

---

## 8. License

MIT License. Built for HH Goa 2026 Shortlisting Task 3.
