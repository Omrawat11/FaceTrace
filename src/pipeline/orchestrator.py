"""Concrete end-to-end FaceTrace pipeline orchestrator.

Wires all backend services into modular pipeline execution:
  Image → Face Detection → Embedding → Search → Candidate Evaluation
  → Match Selection → Evidence Record → Fingerprint → Blockchain Anchor → Verification
"""

import logging
import tempfile
from pathlib import Path

from src.blockchain import get_blockchain_provider
from src.blockchain.base import BlockchainProvider
from src.candidates.base import CandidateMatch, CandidateProcessor
from src.candidates.processor import DefaultCandidateProcessor
from src.core.config import Settings, get_settings
from src.evidence.models import Evidence, EvidenceRecord
from src.face.base import FaceDetector, FaceEmbedder
from src.face.insightface_engine import get_face_detector, get_face_embedder
from src.face.query_processor import (
    QueryFaceProcessingError,
    QueryFaceProcessor,
)
from src.pipeline.base import (
    FaceTracePipeline,
    PipelineOutput,
    PipelineStage,
    ProgressCallback,
)
from src.search import get_search_provider
from src.search.base import SearchProvider
from src.verification.base import VerificationResult
from src.verification.engine import DefaultVerificationEngine, verify_evidence

logger = logging.getLogger(__name__)


class DefaultFaceTracePipeline(FaceTracePipeline):
    """Production pipeline orchestrator executing FaceTrace stages modularly or end-to-end.

    Each service is injected via constructor or defaults to the factory-configured instance.
    The pipeline never stores images or embeddings persistently.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        face_detector: FaceDetector | None = None,
        face_embedder: FaceEmbedder | None = None,
        search_provider: SearchProvider | None = None,
        candidate_processor: CandidateProcessor | None = None,
        blockchain_provider: BlockchainProvider | None = None,
        progress_callback: ProgressCallback = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._detector = face_detector or get_face_detector(self._settings)
        self._embedder = face_embedder or get_face_embedder(self._settings)
        self._search = search_provider or get_search_provider(self._settings)
        self._candidates = candidate_processor or DefaultCandidateProcessor(
            face_detector=self._detector,
            face_embedder=self._embedder,
        )
        self._blockchain = blockchain_provider or get_blockchain_provider(self._settings)
        self._progress = progress_callback
        self._threshold = self._settings.FACE_SIMILARITY_THRESHOLD
        self._max_candidates = getattr(
            self._settings, "MAX_CANDIDATES", getattr(self._settings, "MAX_SEARCH_RESULTS", 20)
        )

    def _notify(self, stage: PipelineStage, message: str) -> None:
        """Send progress notification to the UI callback if registered."""
        if self._progress:
            try:
                self._progress(stage.value, message)
            except Exception:
                pass  # Never let callback failures break the pipeline

    def search_and_match(self, query_image: bytes | Path | str) -> PipelineOutput:
        """Execute Stages 1 to 4: Detection, Biometric Encoding, Web Search, and Candidate Evaluation.

        Does not automatically anchor to the blockchain, allowing user review of discovered candidates.
        """
        if isinstance(query_image, bytes):
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(query_image)
            tmp.close()
            image_path = Path(tmp.name)
            image_bytes = query_image
        else:
            image_path = Path(str(query_image))
            image_bytes = image_path.read_bytes()

        output = PipelineOutput(
            query_image_path=image_path,
            stage=PipelineStage.IDLE,
        )

        # ──────────────────────────────────────────────
        # Stage 1: Face Detection + Validation
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.FACE_DETECTION
        self._notify(output.stage, "Detecting and validating face in query image...")

        try:
            qfp = QueryFaceProcessor(detector=self._detector, embedder=self._embedder)
            detected_face, embedding = qfp.process_query_image(image_bytes)
        except QueryFaceProcessingError as exc:
            output.stage = PipelineStage.FAILED
            output.error_message = str(exc)
            self._notify(output.stage, f"Face validation failed: {exc}")
            return output
        except Exception as exc:
            output.stage = PipelineStage.FAILED
            output.error_message = f"Unexpected error during face detection: {exc}"
            self._notify(output.stage, output.error_message)
            return output

        output.query_face = detected_face
        output.stage = PipelineStage.FACE_ENCODING
        self._notify(output.stage, "Face validated — 512D ArcFace embedding extracted.")
        output.query_embedding = embedding

        # ──────────────────────────────────────────────
        # Stage 2: Candidate Search
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.SEARCHING
        self._notify(output.stage, f"Sending image to {self._search.provider_name}...")

        try:
            search_results = self._search.search_by_image(
                image_bytes,
                max_results=self._max_candidates,
            )
        except Exception as exc:
            output.stage = PipelineStage.FAILED
            output.error_message = f"Search failed: {exc}"
            self._notify(output.stage, output.error_message)
            return output

        output.search_results = search_results

        if not search_results:
            output.stage = PipelineStage.COMPLETED
            output.error_message = "Search returned zero candidate results from public index."
            self._notify(output.stage, output.error_message)
            return output

        self._notify(output.stage, f"Search results received ({len(search_results)} candidates).")

        # ──────────────────────────────────────────────
        # Stage 3: Candidate Evaluation & Ranking
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.CANDIDATE_EVALUATION
        self._notify(
            output.stage,
            f"Analyzing and extracting faces from {len(search_results)} candidates...",
        )

        try:
            all_matches = self._candidates.fetch_and_evaluate(
                search_results=search_results,
                query_embedding=embedding,
                similarity_threshold=self._threshold,
            )
        except Exception as exc:
            output.stage = PipelineStage.FAILED
            output.error_message = f"Candidate evaluation failed: {exc}"
            self._notify(output.stage, output.error_message)
            return output

        output.all_matches = all_matches

        # ──────────────────────────────────────────────
        # Stage 4: Best Match Selection
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.MATCHING
        self._notify(output.stage, "Ranking matches against query biometric embedding...")

        best_match = self._candidates.select_best_match(all_matches)
        output.best_match = best_match

        if best_match is None:
            output.stage = PipelineStage.COMPLETED
            output.error_message = (
                f"Search completed, but no sufficiently similar face was found (threshold: {self._threshold:.2f}). "
                f"Best similarity score: {all_matches[0].similarity_score:.4f}" if all_matches else
                "No candidates to evaluate."
            )
            self._notify(output.stage, output.error_message)
            return output

        logger.info(
            "Best match selected: '%s' (score=%.4f, threshold=%.2f)",
            best_match.candidate.title,
            best_match.similarity_score,
            self._threshold,
        )

        output.stage = PipelineStage.COMPLETED
        self._notify(output.stage, f"Search & matching complete — {sum(1 for m in all_matches if m.is_match)} match(es) discovered.")
        return output

    def anchor_match(
        self,
        output: PipelineOutput,
        match: CandidateMatch | None = None,
    ) -> PipelineOutput:
        """Execute Stages 5 to 8 for a selected verified CandidateMatch: Evidence, Fingerprint, Anchor, Verify."""
        target_match = match or output.best_match
        if target_match is None:
            output.stage = PipelineStage.FAILED
            output.error_message = "No verified candidate match selected for anchoring."
            self._notify(output.stage, output.error_message)
            return output

        # ──────────────────────────────────────────────
        # Stage 5: Evidence Extraction
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.EVIDENCE_EXTRACTION
        self._notify(output.stage, f"Packaging evidence record for {target_match.candidate.title[:40]}...")

        try:
            evidence = Evidence.create_from_match(match=target_match)
            evidence_record = EvidenceRecord.create_from_match(
                match=target_match,
                threshold=self._threshold,
            )
        except Exception as exc:
            output.stage = PipelineStage.FAILED
            output.error_message = f"Evidence extraction failed: {exc}"
            self._notify(output.stage, output.error_message)
            return output

        output.evidence = evidence
        output.evidence_record = evidence_record

        # ──────────────────────────────────────────────
        # Stage 6: Cryptographic Fingerprinting
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.FINGERPRINTING
        self._notify(output.stage, "Computing SHA-256 fingerprint of canonical evidence...")

        fingerprint = evidence_record.compute_fingerprint()
        bytes32_fp = evidence_record.to_bytes32()
        output.fingerprint = fingerprint

        logger.info("Evidence fingerprint: %s", bytes32_fp)

        # ──────────────────────────────────────────────
        # Stage 7: Blockchain Anchoring
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.BLOCKCHAIN_RECORDING
        self._notify(
            output.stage,
            f"Submitting evidence hash to {self._blockchain.network_name} blockchain...",
        )

        try:
            anchor_result = self._blockchain.anchor_evidence(
                evidence_hash=fingerprint,
                source_url=evidence_record.post_url,
            )
        except Exception as exc:
            output.stage = PipelineStage.FAILED
            output.error_message = f"Blockchain anchoring failed: {exc}"
            self._notify(output.stage, output.error_message)
            return output

        output.anchor_result = anchor_result

        if not anchor_result.is_success:
            output.stage = PipelineStage.FAILED
            output.error_message = f"Blockchain transaction failed: {anchor_result.error or 'Unknown'}"
            self._notify(output.stage, output.error_message)
            return output

        # ──────────────────────────────────────────────
        # Stage 8: On-Chain Verification
        # ──────────────────────────────────────────────
        output.stage = PipelineStage.VERIFICATION
        self._notify(output.stage, "Verifying evidence integrity against on-chain record...")

        try:
            verification = verify_evidence(
                evidence=evidence_record,
                blockchain_provider=self._blockchain,
            )
        except Exception as exc:
            output.stage = PipelineStage.FAILED
            output.error_message = f"Verification failed: {exc}"
            self._notify(output.stage, output.error_message)
            return output

        output.verification_result = verification

        output.stage = PipelineStage.COMPLETED
        status_label = "VERIFIED" if verification.is_verified else verification.status.value
        self._notify(output.stage, f"Pipeline complete — Evidence status: {status_label}")
        logger.info("Pipeline completed successfully. Verification status: %s", status_label)

        return output

    def run(
        self,
        query_image: bytes | Path | str,
        auto_anchor: bool = True,
    ) -> PipelineOutput:
        """Execute the FaceTrace pipeline.

        Args:
            query_image: Raw image bytes, file path, or string path to query face.
            auto_anchor: If True, automatically anchors and verifies the best match if found.
                         If False, stops after candidate ranking for interactive user review.

        Returns:
            PipelineOutput containing intermediate and final artifacts.
        """
        output = self.search_and_match(query_image)
        if output.stage == PipelineStage.FAILED:
            return output

        if auto_anchor and output.best_match is not None:
            output = self.anchor_match(output, output.best_match)

        return output


def get_pipeline(
    settings: Settings | None = None,
    progress_callback: ProgressCallback = None,
) -> DefaultFaceTracePipeline:
    """Factory function creating a fully-configured pipeline instance."""
    return DefaultFaceTracePipeline(
        settings=settings,
        progress_callback=progress_callback,
    )
