"""
agentic_vision_loop.py - Multi-Step Perception-Decision-Action Loop using OpenCV 5
===================================================================================
Implements an autonomous agentic perception workflow where OpenCV 5 visual analysis
directly dictates the subsequent planning, tool invocation, and decision-action path.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .opencv_analyzer import OpenCVAerialAnalyzer, VisionAnalysisResult


@dataclass
class AgenticDecision:
    """Result of an autonomous perception-decision-action cycle."""
    tile_id: str
    timestamp: str
    perception_metrics: Dict[str, Any]
    initial_tabular_score: float
    adjusted_score: float
    decision_branch: str
    recommended_action: str
    tool_calls_invoked: List[str]
    action_payload: Dict[str, Any]
    human_approval_required: bool
    reasoning_trace: List[str] = field(default_factory=list)


class AgenticVisionEngine:
    """
    Autonomous Perception-Decision-Action Agent integrating OpenCV 5
    for urban greenspace planning.
    """

    def __init__(self, analyzer: Optional[OpenCVAerialAnalyzer] = None):
        self.analyzer = analyzer or OpenCVAerialAnalyzer()
        self.decision_history: List[AgenticDecision] = []

    def run_perception_decision_cycle(
        self,
        bgr_image: np.ndarray,
        tile_id: str,
        initial_tabular_score: float = 0.50,
        latitude: float = 28.6139,
        longitude: float = 77.2090,
    ) -> Tuple[AgenticDecision, VisionAnalysisResult]:
        """
        Execute one full Perception-Decision-Action cycle:
          1. Perception: Invoke OpenCV 5 to extract ground-truth visual features.
          2. Synthesis: Compare visual evidence with prior tabular estimates.
          3. Decision: Branch into adaptive pathways based on visual evidence.
          4. Action: Trigger subsequent tool calls, score updates, and action payloads.
        """
        reasoning: List[str] = []
        tools: List[str] = []

        # ── Step 1: PERCEPTION (OpenCV 5 Tool Call) ───────────────────────────
        reasoning.append(f"Initiating visual perception on tile {tile_id} ({latitude:.4f}, {longitude:.4f}).")
        tools.append("opencv5::analyze_aerial_tile")

        vision_result = self.analyzer.analyze_aerial_tile(bgr_image, tile_id=tile_id)
        reasoning.append(
            f"OpenCV 5 extracted visual evidence: Canopy={vision_result.canopy_fraction*100:.1f}%, "
            f"BareGround={vision_result.bare_ground_fraction*100:.1f}%, "
            f"Impervious={vision_result.impervious_fraction*100:.1f}%, "
            f"LargestParcel={vision_result.largest_bare_parcel_sqm:.0f} m²."
        )

        # ── Step 2 & 3: DECISION (Visual Evidence alters plan & next actions) ──
        adjusted_score = initial_tabular_score
        action_payload: Dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "canopy_detected_pct": vision_result.canopy_fraction * 100.0,
            "bare_ground_pct": vision_result.bare_ground_fraction * 100.0,
        }

        # Branch 1: Hard Water / River Exclusion
        if vision_result.water_fraction >= 0.15:
            decision_branch = "BRANCH_WATER_EXCLUSION"
            adjusted_score = 0.0
            recommended_action = "Enforce Strict Ecological Water Body Preservation; Exclude Ground Planting."
            tools.append("planning::enforce_water_buffer_ordinance")
            reasoning.append("Visual evidence confirms active surface water bodies (>=15%). Zeroing out afforestation score.")
            action_payload["protection_type"] = "Riverine / Wetland Sanctuary"
            human_approval_required = False

        # Branch 2: Ground-Truth Canopy Verified (Discrepancy Resolution)
        elif vision_result.canopy_fraction >= 0.30:
            decision_branch = "BRANCH_CANOPY_PROTECTION"
            # Reduce urgency since canopy is already healthy on the ground
            adjusted_score = round(initial_tabular_score * 0.45, 4)
            recommended_action = "Canopy Verified: Discrepancy Resolved. Shift from Afforestation to Canopy Protection."
            tools.append("forestry::issue_canopy_conservation_order")
            tools.append("monitoring::schedule_quarterly_satellite_audit")
            reasoning.append(
                f"Visual inspection discovered {vision_result.canopy_fraction*100:.1f}% existing healthy tree canopy "
                "missed by low-resolution rasters. Re-routing action to conservation monitoring."
            )
            action_payload["protection_tier"] = "Existing Urban Canopy"
            action_payload["re_inspection_interval_days"] = 90
            human_approval_required = False

        # Branch 3: High-Yield Miyawaki Micro-Forest Candidate
        elif vision_result.largest_bare_parcel_sqm >= 1000.0:
            decision_branch = "BRANCH_MIYAWAKI_MICRO_FOREST"
            # Boost score by +40% (opportunity bonus)
            adjusted_score = min(1.0, round(initial_tabular_score * 1.40, 4))
            # 3 saplings per square meter under high-density Miyawaki methodology
            estimated_trees = int(vision_result.largest_bare_parcel_sqm * 3.0)
            recommended_action = f"Prime Miyawaki Site Approved: Deploy Dense Micro-Forest ({estimated_trees:,} saplings)."
            tools.append("landscape::generate_miyawaki_planting_blueprint")
            tools.append("nursery::reserve_native_species_saplings")
            reasoning.append(
                f"OpenCV contour analysis identified continuous unpaved bare soil parcel of "
                f"{vision_result.largest_bare_parcel_sqm:,.0f} m². Escalating to high-priority micro-forest intervention."
            )
            action_payload["estimated_sapling_capacity"] = estimated_trees
            action_payload["recommended_species"] = ["Neem", "Peepal", "Amaltas", "Jamun", "Dhau"]
            action_payload["parcel_area_sqm"] = vision_result.largest_bare_parcel_sqm
            human_approval_required = True  # Requires municipal sign-off

        # Branch 4: Hyper-Dense Concrete Jungle -> Rooftop Greening
        elif vision_result.impervious_fraction >= 0.75 and vision_result.bare_ground_fraction < 0.10:
            decision_branch = "BRANCH_ROOFTOP_CORRIDOR"
            adjusted_score = round(initial_tabular_score * 1.15, 4)
            recommended_action = "Ground Space Constrained: Trigger Rooftop Garden & Vertical Greening Corridor Program."
            tools.append("urban_planning::initiate_rooftop_subsidy_program")
            tools.append("gis::extract_building_rooftop_geometries")
            reasoning.append(
                f"OpenCV visual analysis detected {vision_result.impervious_fraction*100:.1f}% impervious concrete with "
                f"<10% bare soil. Ground planting impossible. Dynamically re-routing strategy to structural rooftops."
            )
            action_payload["intervention_mode"] = "Green Roofs & Vertical Living Walls"
            action_payload["incentive_tier"] = "Municipal Property Tax Rebate Eligible"
            human_approval_required = False

        # Branch 5: Moderate Open Ground / Linear Avenue Planting
        else:
            decision_branch = "BRANCH_LINEAR_AVENUE_PLANTING"
            adjusted_score = initial_tabular_score
            recommended_action = "Linear Street Tree Canopy & Pocket Park Installation."
            tools.append("pwd::schedule_roadside_avenue_planting")
            reasoning.append("Moderate urban fragmentation detected. Recommending linear roadside tree planting.")
            action_payload["intervention_mode"] = "Linear Roadside Corridor"
            human_approval_required = False

        # ── Step 4: ACTION & RECORDING ────────────────────────────────────────
        decision = AgenticDecision(
            tile_id=tile_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            perception_metrics={
                "canopy_pct": vision_result.canopy_fraction * 100.0,
                "bare_ground_pct": vision_result.bare_ground_fraction * 100.0,
                "impervious_pct": vision_result.impervious_fraction * 100.0,
                "water_pct": vision_result.water_fraction * 100.0,
                "edge_density": vision_result.edge_density,
                "largest_parcel_sqm": vision_result.largest_bare_parcel_sqm,
                "dominant_class": vision_result.dominant_class,
            },
            initial_tabular_score=initial_tabular_score,
            adjusted_score=adjusted_score,
            decision_branch=decision_branch,
            recommended_action=recommended_action,
            tool_calls_invoked=tools,
            action_payload=action_payload,
            human_approval_required=human_approval_required,
            reasoning_trace=reasoning,
        )

        self.decision_history.append(decision)
        return decision, vision_result
