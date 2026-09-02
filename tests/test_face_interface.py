"""Unit tests for face detection data models and cosine similarity calculation."""

import pytest

from src.face.base import BoundingBox, compute_cosine_similarity


def test_bounding_box_properties() -> None:
    """Test width, height, area, and list representation of BoundingBox."""
    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=170)
    assert bbox.width == 100
    assert bbox.height == 150
    assert bbox.area == 15000
    assert bbox.as_list() == [10, 20, 110, 170]


def test_cosine_similarity_identical_vectors() -> None:
    """Test that identical vectors have a cosine similarity of 1.0."""
    v1 = [0.5, 0.5, 0.5, 0.5]
    assert pytest.approx(compute_cosine_similarity(v1, v1), rel=1e-5) == 1.0


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Test that orthogonal vectors have a cosine similarity of 0.0."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert pytest.approx(compute_cosine_similarity(v1, v2), abs=1e-5) == 0.0


def test_cosine_similarity_opposite_vectors() -> None:
    """Test that opposite vectors have a cosine similarity of -1.0."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [-1.0, 0.0, 0.0]
    assert pytest.approx(compute_cosine_similarity(v1, v2), rel=1e-5) == -1.0


def test_cosine_similarity_zero_vector() -> None:
    """Test that zero vectors return 0.0 safely without ZeroDivisionError."""
    v1 = [0.0, 0.0, 0.0]
    v2 = [1.0, 2.0, 3.0]
    assert compute_cosine_similarity(v1, v2) == 0.0


def test_cosine_similarity_dimension_mismatch() -> None:
    """Test that differing vector dimensions raise ValueError."""
    v1 = [1.0, 2.0]
    v2 = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        compute_cosine_similarity(v1, v2)
