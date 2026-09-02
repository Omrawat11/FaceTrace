"""End-to-End demonstration script for FaceTrace Phase 3: Evidence Fingerprinting & Blockchain Anchoring."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.blockchain.service import EthereumBlockchainService
from src.candidates.processor import DefaultCandidateProcessor
from src.core.config import get_settings
from src.core.types import VerificationStatus
from src.evidence.models import EvidenceRecord
from src.face.insightface_engine import InsightFaceDetector, InsightFaceEmbedder
from src.face.query_processor import QueryFaceProcessor
from src.search.mock import MockSearchProvider
from src.verification.engine import DefaultVerificationEngine

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_phase3_demo(query_image_path: str = "data/demo_images/user_primary.jpg") -> None:
    settings = get_settings()
    query_path = Path(query_image_path)

    print("================================================================================")
    print("FACETRACE PHASE 3: EVIDENCE FINGERPRINTING & LOCAL BLOCKCHAIN ANCHORING")
    print("================================================================================")
    print(f"Network Mode: LOCAL EVM (In-Process Ethereum VM)")
    print(f"Query Image: {query_path.resolve()}\n")

    # Step 1: Query Face Processing
    print("[1/7] Processing Query Face...")
    detector = InsightFaceDetector(confidence_threshold=settings.FACE_DETECTION_CONFIDENCE)
    embedder = InsightFaceEmbedder()
    query_processor = QueryFaceProcessor(detector=detector, embedder=embedder)

    detected_face, query_embedding = query_processor.process_query_image(query_path)
    print(f"  [+] Exactly 1 face validated (BBox: {detected_face.bbox.as_list()}, Conf: {detected_face.confidence:.4f})")
    print(f"  [+] Extracted 512D ArcFace biometric embedding.\n")

    # Step 2: Local Search Discovery
    print("[2/7] Local Search Discovery (MockSearchProvider)...")
    search_provider = MockSearchProvider(demo_dir=settings.DEMO_IMAGES_DIR)
    search_results = search_provider.search_by_image(query_path, max_results=10)
    print(f"  [+] Discovered {len(search_results)} candidates across public indexed platforms.\n")

    # Step 3: Candidate Processing & Ranking
    print("[3/7] Candidate Face Processing & Biometric Ranking...")
    candidate_processor = DefaultCandidateProcessor(
        face_detector=detector,
        face_embedder=embedder,
    )
    matches = candidate_processor.fetch_and_evaluate(
        search_results=search_results,
        query_embedding=query_embedding,
        similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD,
    )

    best_match = candidate_processor.select_best_match(matches)
    if not best_match:
        print("  [-] No matching candidate found exceeding threshold. Cannot anchor evidence.")
        return

    print(f"  [+] Best Match Selected: {best_match.candidate.title}")
    print(f"      - Source URL: {best_match.candidate.url}")
    print(f"      - Platform: {best_match.candidate.platform.value.upper()}")
    print(f"      - Biometric Similarity: {best_match.similarity_score:.4f} (Threshold: {settings.FACE_SIMILARITY_THRESHOLD:.2f})\n")

    # Step 4: Evidence Packaging & Dual Cryptographic Fingerprinting
    print("[4/7] Evidence Packaging & Cryptographic Fingerprinting...")
    evidence_record = EvidenceRecord.create_from_match(
        match=best_match,
        threshold=settings.FACE_SIMILARITY_THRESHOLD,
        caption="Verified public matching social portrait.",
    )

    image_sha256 = evidence_record.image_sha256
    evidence_sha256 = evidence_record.compute_fingerprint()
    evidence_bytes32 = evidence_record.to_bytes32()

    print(f"  [+] Image SHA-256:    {image_sha256}")
    print(f"      (Fingerprint of exact candidate raw image bytes)")
    print(f"  [+] Evidence SHA-256: {evidence_sha256}")
    print(f"      (Fingerprint of RFC 8785 canonical evidence JSON)")
    print(f"  [+] EVM bytes32:      {evidence_bytes32}")
    print(f"  [+] Canonical Evidence JSON Payload:")
    print(f"      {evidence_record.to_canonical_json()}\n")

    # Step 5: Local Blockchain Deployment & Anchoring
    print("[5/7] Anchoring Evidence Fingerprint to Smart Contract...")
    blockchain_service = EthereumBlockchainService(network_name="local")
    if not blockchain_service.contract_address:
        print("  [+] Deploying EvidenceRegistry smart contract to local EVM...")
        contract_addr = blockchain_service.deploy_contract()
        print(f"      Contract Address: {contract_addr}")
    else:
        print(f"  [+] Using EvidenceRegistry Contract at: {blockchain_service.contract_address}")

    anchor_res = blockchain_service.anchor_evidence(
        evidence_hash=evidence_bytes32,
        source_url=evidence_record.post_url,
    )

    if not anchor_res.success:
        print(f"  [-] Failed to anchor evidence on-chain: {anchor_res.error}")
        return

    print(f"  [+] Transaction Submitted & Confirmed!")
    print(f"      - Transaction Hash: {anchor_res.tx_hash}")
    print(f"      - Block Number:     #{anchor_res.block_number}")
    print(f"      - Contract Address: {anchor_res.contract_address}")
    print(f"      - Status:           {'SUCCESS (1)' if anchor_res.is_success else 'FAILED (0)'}\n")

    # Step 6: On-Chain Record Retrieval & Verification
    print("[6/7] Querying On-Chain Evidence Record & Cryptographic Verification...")
    on_chain_record = blockchain_service.get_evidence(evidence_bytes32)
    hashes_match = on_chain_record.fingerprint.lower() == evidence_bytes32.lower()

    verifier = DefaultVerificationEngine(blockchain_provider=blockchain_service)
    verification_res = verifier.verify(evidence_record)

    print(f"  [+] On-Chain Fingerprint Retrieved: {on_chain_record.fingerprint}")
    print(f"  [+] Local vs On-Chain Match:        {'MATCH' if hashes_match else 'MISMATCH'}")
    print(f"  [+] On-Chain Verification Status:   {verification_res.status.value}")
    print(f"      - Verified?:        {verification_res.is_verified}")
    print(f"      - Recorded By:      {on_chain_record.recorder}")
    print(f"      - Block Timestamp:  {on_chain_record.formatted_time}")
    print(f"      - Mined In Block:   #{on_chain_record.block_number}")
    print(f"      - Anchored URL:     {on_chain_record.source_url}\n")

    # Step 7: Tamper Detection Demonstration
    print("[7/7] Demonstrating Tamper-Evidence Detection...")
    # Modify exactly one field: similarity_score (e.g. 1.0000 -> 0.8300)
    original_score = evidence_record.similarity_score
    modified_score = round(original_score - 0.1700, 4) if original_score >= 0.5 else round(original_score + 0.1700, 4)

    tampered_record = EvidenceRecord(
        post_url=evidence_record.post_url,
        platform=evidence_record.platform,
        title=evidence_record.title,
        image_sha256=evidence_record.image_sha256,
        similarity_score=modified_score,
        threshold=evidence_record.threshold,
        discovered_at=evidence_record.discovered_at,
        evidence_version=evidence_record.evidence_version,
        caption=evidence_record.caption,
    )

    evidence_hash_a = evidence_bytes32
    evidence_hash_b = tampered_record.to_bytes32()

    # Compare directly: Evidence Hash B != Blockchain Hash A
    hashes_differ = (evidence_hash_b.lower() != on_chain_record.fingerprint.lower())
    tamper_result_label = "TAMPERED / INVALID" if hashes_differ else "UNEXPECTED MATCH"

    # Verifier check comparing tampered record with original anchored fingerprint
    tamper_check = verifier.verify(tampered_record, expected_fingerprint=evidence_hash_a)

    print(f"  [+] Modified field: similarity_score ({original_score:.4f} -> {modified_score:.4f})")
    print(f"  [+] Original Hash A:    {evidence_hash_a}")
    print(f"  [+] On-Chain Hash:      {on_chain_record.fingerprint}")
    print(f"  [+] Modified Hash B:    {evidence_hash_b}")
    print(f"  [+] Direct Comparison:  Evidence Hash B != Blockchain Hash A ({hashes_differ})")
    print(f"  [+] Tamper Status:      {tamper_check.status.value}")
    print(f"      - Is Verified?:     {tamper_check.is_verified}")
    print(f"      - Outcome:          {tamper_result_label}")
    print(f"      - Details:          {tamper_check.tamper_reasons}\n")

    print("================================================================================")
    print("PHASE 3 FINAL SUMMARY REPORT")
    print("================================================================================")
    print(f"IMAGE SHA-256:\n{image_sha256}\n")
    print(f"EVIDENCE SHA-256:\n{evidence_sha256}\n")
    print(f"CONTRACT:\n{anchor_res.contract_address}\n")
    print(f"TRANSACTION:\n{anchor_res.tx_hash}\n")
    print(f"BLOCK:\n#{anchor_res.block_number}\n")
    print(f"ON-CHAIN HASH:\n{on_chain_record.fingerprint}\n")
    print(f"VERIFICATION:\n{'✓ VERIFIED' if verification_res.is_verified else '✗ FAILED'}\n")
    print("TAMPER TEST:\n")
    print(f"ORIGINAL HASH:\n{evidence_hash_a}\n")
    print(f"MODIFIED HASH:\n{evidence_hash_b}\n")
    print(f"RESULT:\n{'✗ TAMPERED' if not tamper_check.is_verified else '✓ UNTAMPERED'}")
    print("================================================================================")


if __name__ == "__main__":
    run_phase3_demo()

