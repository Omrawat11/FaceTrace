"""End-to-End demonstration script for FaceTrace Phase 2B Local Pipeline."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.candidates.processor import DefaultCandidateProcessor
from src.core.config import get_settings
from src.face.insightface_engine import InsightFaceDetector, InsightFaceEmbedder
from src.face.query_processor import QueryFaceProcessor
from src.search.mock import MockSearchProvider


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_phase2b_demo(query_image_path: str = "data/demo_images/user_primary.jpg") -> None:
    settings = get_settings()
    query_path = Path(query_image_path)

    print("================================================================================")
    print("FACETRACE PHASE 2B: LOCAL SEARCH & REAL BIOMETRIC FACE MATCHING PIPELINE")
    print("================================================================================")
    print(f"Configured Similarity Threshold: {settings.FACE_SIMILARITY_THRESHOLD:.2f}")
    print(f"Query Image Path: {query_path.resolve()}\n")

    # Step 1: Query Face Processing
    print("[1/5] Processing Query Image...")
    detector = InsightFaceDetector(confidence_threshold=settings.FACE_DETECTION_CONFIDENCE)
    embedder = InsightFaceEmbedder()
    query_processor = QueryFaceProcessor(detector=detector, embedder=embedder)

    detected_face, query_embedding = query_processor.process_query_image(query_path)
    print(f"  [+] Validated image format.")
    print(f"  [+] Exactly 1 face detected.")
    print(f"      - Bounding Box: {detected_face.bbox.as_list()}")
    print(f"      - Detection Confidence: {detected_face.confidence:.4f}")
    print(f"  [+] Extracted 512D ArcFace biometric embedding vector (norm={sum(v*v for v in query_embedding.vector)**0.5:.4f})\n")

    # Step 2: Local Search Discovery via MockSearchProvider
    print("[2/5] Querying Local Search Provider (MockSearchProvider)...")
    search_provider = MockSearchProvider(demo_dir=settings.DEMO_IMAGES_DIR)
    search_results = search_provider.search_by_image(query_path, max_results=10)
    print(f"  [+] Search Provider: {search_provider.provider_name.upper()}")
    print(f"  [+] Discovered {len(search_results)} candidate results across indexed platforms:\n")
    for idx, res in enumerate(search_results, 1):
        print(f"    {idx}. [{res.platform.value.upper():9s}] {res.title}")
        print(f"       URL: {res.url}")

    # Step 3: Candidate Retrieval, Detection & Biometric Evaluation
    print("\n[3/5] Evaluating Candidate Faces & Computing Cosine Similarities...")
    candidate_processor = DefaultCandidateProcessor(
        face_detector=detector,
        face_embedder=embedder,
    )
    matches = candidate_processor.fetch_and_evaluate(
        search_results=search_results,
        query_embedding=query_embedding,
        similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD,
    )

    # Step 4: Candidate Ranking
    print("\n[4/5] Candidate Ranking (similarity_score DESC):")
    print("--------------------------------------------------------------------------------")
    print(f"{'Rank':<5} {'Score':<8} {'Status':<18} {'Faces':<7} {'Platform':<10} {'Title'}")
    print("--------------------------------------------------------------------------------")
    for rank, m in enumerate(matches, 1):
        score_str = f"{m.similarity_score:.4f}"
        faces_str = str(m.detected_faces_count)
        platform_str = m.candidate.platform.value.upper()
        title_str = (m.candidate.title[:38] + "...") if len(m.candidate.title) > 40 else m.candidate.title
        print(f"{rank:<5} {score_str:<8} {m.status:<18} {faces_str:<7} {platform_str:<10} {title_str}")
    print("--------------------------------------------------------------------------------")

    # Step 5: Best Match Selection
    best_match = candidate_processor.select_best_match(matches)
    print("\n[5/5] Best Match Selection:")
    if best_match:
        print(f"  [+] FINAL RESULT: MATCH FOUND")
        print(f"  [+] Top Matching Candidate: {best_match.candidate.title}")
        print(f"  [+] Source URL: {best_match.candidate.url}")
        print(f"  [+] Platform: {best_match.candidate.platform.value.upper()}")
        print(f"  [+] Candidate Image SHA-256: {best_match.candidate.image_hash}")
        print(f"  [+] Similarity Score: {best_match.similarity_score:.4f} (Threshold: {settings.FACE_SIMILARITY_THRESHOLD:.2f})")
    else:
        print(f"  [-] FINAL RESULT: NO MATCHING CANDIDATE EXCEEDED THRESHOLD ({settings.FACE_SIMILARITY_THRESHOLD:.2f})")

    print("\n================================================================================")
    print("DEMO RUN COMPLETED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    run_phase2b_demo()
