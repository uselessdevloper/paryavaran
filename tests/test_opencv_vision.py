"""
test_opencv_vision.py - Comprehensive Test Suite for OpenCV 5 Agentic Vision
=============================================================================
Validates OpenCV 5 computer vision algorithms, segmentation accuracy,
and agentic branching logic across diverse urban micro-zone scenarios.
"""

import os
import sys
import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vision_engine.opencv_analyzer import OpenCVAerialAnalyzer
from vision_engine.agentic_vision_loop import AgenticVisionEngine
from vision_engine.generate_sample_tiles import generate_sample_tiles


@pytest.fixture(scope="session")
def sample_tile_paths():
    """Ensure sample tiles exist and return their filepaths."""
    return generate_sample_tiles()


@pytest.fixture
def analyzer():
    return OpenCVAerialAnalyzer(tile_size_meters=250.0)


@pytest.fixture
def engine(analyzer):
    return AgenticVisionEngine(analyzer=analyzer)


def test_opencv5_version():
    """Verify OpenCV version is 5.x or 4.x compliant."""
    major_version = int(cv2.__version__.split(".")[0])
    assert major_version >= 4, f"Expected OpenCV 4.x or 5.x, got {cv2.__version__}"


def test_contrast_enhancement_clahe(analyzer):
    """Test atmospheric haze reduction using CLAHE."""
    dummy_hazy = np.full((100, 100, 3), 180, dtype=np.uint8)
    enhanced = analyzer.enhance_contrast_and_dehaze(dummy_hazy)
    assert enhanced.shape == dummy_hazy.shape
    assert enhanced.dtype == np.uint8


def test_dense_canopy_branch(engine, sample_tile_paths):
    """Verify that a forest canopy tile triggers BRANCH_CANOPY_PROTECTION and score reduction."""
    img = cv2.imread(sample_tile_paths["existing_canopy"])
    assert img is not None

    decision, vision = engine.run_perception_decision_cycle(
        bgr_image=img,
        tile_id="TEST_CANOPY",
        initial_tabular_score=0.60,
    )

    assert vision.canopy_fraction >= 0.50
    assert decision.decision_branch == "BRANCH_CANOPY_PROTECTION"
    assert decision.adjusted_score < decision.initial_tabular_score
    assert "forestry::issue_canopy_conservation_order" in decision.tool_calls_invoked


def test_bare_opportunity_branch(engine, sample_tile_paths):
    """Verify that unpaved vacant ground triggers BRANCH_MIYAWAKI_MICRO_FOREST and score boost."""
    img = cv2.imread(sample_tile_paths["bare_opportunity"])
    assert img is not None

    decision, vision = engine.run_perception_decision_cycle(
        bgr_image=img,
        tile_id="TEST_BARE",
        initial_tabular_score=0.65,
    )

    assert vision.bare_ground_fraction >= 0.25
    assert vision.largest_bare_parcel_sqm >= 1000.0
    assert decision.decision_branch == "BRANCH_MIYAWAKI_MICRO_FOREST"
    assert decision.adjusted_score > decision.initial_tabular_score
    assert "landscape::generate_miyawaki_planting_blueprint" in decision.tool_calls_invoked
    assert decision.action_payload["estimated_sapling_capacity"] > 0


def test_dense_concrete_branch(engine, sample_tile_paths):
    """Verify that a concrete urban matrix triggers BRANCH_ROOFTOP_CORRIDOR."""
    img = cv2.imread(sample_tile_paths["dense_concrete"])
    assert img is not None

    decision, vision = engine.run_perception_decision_cycle(
        bgr_image=img,
        tile_id="TEST_CONCRETE",
        initial_tabular_score=0.72,
    )

    assert vision.impervious_fraction >= 0.75
    assert decision.decision_branch == "BRANCH_ROOFTOP_CORRIDOR"
    assert "urban_planning::initiate_rooftop_subsidy_program" in decision.tool_calls_invoked


def test_water_exclusion_branch(engine, sample_tile_paths):
    """Verify that water surface triggers BRANCH_WATER_EXCLUSION and score zeroing."""
    img = cv2.imread(sample_tile_paths["yamuna_river"])
    assert img is not None

    decision, vision = engine.run_perception_decision_cycle(
        bgr_image=img,
        tile_id="TEST_WATER",
        initial_tabular_score=0.55,
    )

    assert decision.decision_branch == "BRANCH_WATER_EXCLUSION"
    assert decision.adjusted_score == 0.0
    assert "planning::enforce_water_buffer_ordinance" in decision.tool_calls_invoked


if __name__ == "__main__":
    pytest.main(["-v", __file__])
