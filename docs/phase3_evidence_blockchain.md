# FaceTrace Phase 3: Evidence Fingerprinting & Blockchain Anchoring Guide

> **Judges & Evaluators Quick Reference**: This document explains how FaceTrace converts a verified biometric match into an immutable, tamper-evident cryptographic fingerprint anchored onto an Ethereum-compatible blockchain.

---

## 1. What an EvidenceRecord Is

An `EvidenceRecord` is a lightweight, structured data object representing a verified candidate face discovered across public web/social platforms. It bundles **only** the essential non-biometric provenance metadata needed to prove when, where, and how confidently a match was discovered:

```json
{
  "post_url": "https://instagram.com/p/C_abc123xyz/",
  "platform": "instagram",
  "title": "Jane Doe (@janedoe) • Instagram portrait photo (Match 01)",
  "image_sha256": "3fd9c94c5ea2ba4cbf8aa531502f7f1da450901f3f45225ab84fd3b6c95399a6",
  "similarity_score": 1.0,
  "threshold": 0.45,
  "discovered_at": "2026-09-02T16:32:39Z",
  "evidence_version": "1.0.0",
  "caption": "Verified public matching social portrait."
}
```

---

## 2. Image SHA-256 vs. Evidence SHA-256

FaceTrace enforces a strict conceptual and technical separation between the two hashes:

| Property | Image SHA-256 (`image_sha256`) | Evidence SHA-256 (`evidence_sha256`) |
| :--- | :--- | :--- |
| **Input** | The exact, raw binary bytes of the candidate image file. | The canonical RFC 8785 JSON representation of the entire `EvidenceRecord`. |
| **Formula** | `SHA256(raw_image_bytes)` | `SHA256(canonicalize_evidence_dict(record))` |
| **Purpose** | Proves whether the candidate photo file has been modified, cropped, or compressed. | Cryptographically binds URLs, platforms, similarity scores, timestamps, and the image hash into an unforgeable 32-byte identity. |
| **Storage** | Stored inside the `EvidenceRecord`. | Formatted as EVM `bytes32` (`0x...`) and anchored on the blockchain. |

---

## 3. Canonicalization (RFC 8785)

In JSON, two dictionaries containing identical data might have different key orderings or whitespace formatting:
- Record A: `{"platform":"instagram","similarity_score":0.82}`
- Record B: `{"similarity_score":0.82,"platform":"instagram"}`

Without canonicalization, Record A and Record B would yield different SHA-256 hashes!

FaceTrace solves this using the **RFC 8785 Canonical JSON** standard:
1. **Sorted Keys**: Keys are strictly sorted lexicographically (`discovered_at`, `evidence_version`, `image_sha256`, `platform`, `post_url`, `similarity_score`, `threshold`, `title`).
2. **Whitespace Stripping**: Minimal delimiters (`:`, `,`) with zero extraneous spaces or newlines.
3. **Unicode NFC**: Text strings are normalized to Unicode Canonical Composition (NFC).
4. **Float Stability**: Floating-point scores are normalized to 4 decimal places (`0.8200`), avoiding floating-point precision drift across CPU architectures.
5. **Deterministic Output**: Guarantees that any system in the world parsing the same logical evidence produces the exact same string of bytes.

---

## 4. Why Hashing Is Deterministic

Cryptographic hashing with SHA-256 is an invariant mathematical function:
- **Same Input $\rightarrow$ Exact Same Hash**: For the exact same binary input, SHA-256 always produces the exact same 256-bit (64 hex character) digest.
- **The Avalanche Effect**: Modifying even a single bit (e.g. changing `similarity_score` from `0.8200` to `0.8201`) causes over 50% of the output bits to change unpredictably.
- **Preimage Resistance**: It is computationally impossible to reverse-engineer the original image or metadata from the hash.

---

## 5. What the Smart Contract Stores

The `EvidenceRegistry.sol` smart contract is an immutable, append-only registry that stores:
- `bytes32 fingerprint`: The 32-byte SHA-256 evidence fingerprint.
- `address recorder`: The EVM account address that submitted the transaction.
- `uint256 timestamp`: The block timestamp when the transaction was mined into the blockchain.
- `uint256 blockNumber`: The block number containing the transaction.
- `string sourceUrl`: The public web/social URL where the evidence was discovered.

---

## 6. What the Blockchain Does NOT Store (Zero Biometrics On-Chain)

To uphold user privacy and legal compliance (GDPR / CCPA / BIPA):
- ❌ **NO Raw Images**: The user's query photo and the candidate photo are never sent to or stored on-chain.
- ❌ **NO Face Embeddings**: High-dimensional biometric vectors (e.g. 512D ArcFace embeddings) never touch the blockchain.
- ❌ **NO Personal Identifiable Information (PII)**: No names, phone numbers, email addresses, or government IDs.
- ❌ **NO API Keys**: All secrets are kept local and excluded from records.

---

## 7. Local Blockchain Setup

FaceTrace uses an Ethereum-compatible environment that runs locally with zero configuration and zero financial cost:
- **In-Process EVM (`EthereumTesterProvider`)**: Automatically initializes a real, in-memory Ethereum Virtual Machine powered by `eth-tester` and `py-evm`. No background daemons or external tools required.
- **Local JSON-RPC Nodes**: Automatically connects to Anvil or Hardhat if running on `http://127.0.0.1:8545`.
- **Mock Provider (`MockBlockchainProvider`)**: An in-memory provider available for ultra-fast unit testing.

---

## 8. Contract Deployment

Deploying `EvidenceRegistry.sol` is fully automated and reproducible:
```bash
python scripts/deploy_contract.py
```
Flow:
1. Loads compiled bytecode and ABI from `contracts/EvidenceRegistry.json`.
2. Connects to the local EVM provider.
3. Deploys the contract constructor from the active deployer account.
4. Waits for the deployment receipt and confirms `status == 1`.
5. Writes deployment metadata (`contractAddress`, `deployer`, `timestamp`) to `contracts/deployment_local.json`.

---

## 9. Evidence Anchoring (`anchor_evidence`)

The application anchors evidence through `anchor_evidence()`:
```python
from src.blockchain import anchor_evidence

result = anchor_evidence(
    evidence_hash=evidence_record.to_bytes32(),
    source_url=evidence_record.post_url,
)
```
Behind the scenes:
1. Formats the 64-char fingerprint into a 32-byte hex string (`0x...`).
2. Submits the `recordEvidence(bytes32, string)` transaction to the smart contract.
3. Waits for the mined transaction receipt.
4. Validates receipt status (`status == 1`).
5. Queries the contract to confirm the stored fingerprint matches the submitted fingerprint.
6. Returns an `AnchorResult` containing `tx_hash`, `block_number`, `contract_address`, `stored_evidence_hash`, and `timestamp`.

---

## 10. Evidence Verification (`verify_evidence`)

To independently verify an evidence bundle:
```python
from src.verification import verify_evidence

result = verify_evidence(evidence_record)
```
Verification Workflow:
1. Deterministically recomputes the canonical JSON and SHA-256 fingerprint from the evidence record.
2. Calls `getEvidence(bytes32)` on the smart contract (a zero-gas view call).
3. If the record does not exist $\rightarrow$ returns `VerificationStatus.NOT_FOUND`.
4. If on-chain metadata (e.g. `sourceUrl`) differs $\rightarrow$ returns `VerificationStatus.TAMPERED`.
5. If the on-chain hash matches $\rightarrow$ returns `VerificationStatus.VERIFIED`.
6. If the RPC connection fails $\rightarrow$ returns `VerificationStatus.BLOCKCHAIN_ERROR`.

---

## 11. Tamper Detection

Because the blockchain anchors the exact SHA-256 hash of the canonical evidence:
- If a bad actor alters **even one field** (e.g., claiming a similarity score was `0.95` instead of `0.82`, or tampering with the source URL):
  1. The canonical JSON changes.
  2. The recalculated Evidence Hash differs completely.
  3. The modified hash is not present in the blockchain registry.
  4. The system flags the evidence as **`TAMPERED / INVALID`**.
- This guarantees end-to-end cryptographic provenance that holds up in legal and investigative audits.
