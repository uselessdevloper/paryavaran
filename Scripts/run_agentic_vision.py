"""
run_agentic_vision.py - Standalone CLI Runner & Evaluation Script for OpenCV 5
================================================================================
Demonstrates the autonomous Perception-Decision-Action loop across Delhi micro-zones,
generates OpenCV 5 visual annotations, and exports an agentic decision trace.
"""

import json
import os
import sys
import cv2

# Add root folder to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from vision_engine.agentic_vision_loop import AgenticVisionEngine
from vision_engine.generate_sample_tiles import generate_sample_tiles


def run_demonstration():
    print("=" * 80)
    print("🌿 PARYAVARAN: OpenCV 5 Agentic Aerial Perception & Canopy Verification Engine")
    print(f"OpenCV Version: {cv2.__version__}")
    print("=" * 80)

    # 1. Ensure sample tiles exist
    tiles = generate_sample_tiles()
    output_dir = os.path.join(ROOT, "vision_engine/output_inspections")
    os.makedirs(output_dir, exist_ok=True)

    engine = AgenticVisionEngine()

    test_cases = [
        ("delhi_bare_opportunity.png", "ZONE_OKHLA_BARE", 0.65, 28.5355, 77.2732),
        ("delhi_dense_concrete.png", "ZONE_CP_DENSE", 0.72, 28.6315, 77.2167),
        ("delhi_existing_canopy.png", "ZONE_RIDGE_CANOPY", 0.60, 28.6012, 77.1850),
        ("delhi_yamuna_river.png", "ZONE_YAMUNA_RIVER", 0.55, 28.6450, 77.2580),
    ]

    all_traces = []

    for filename, tile_id, initial_score, lat, lon in test_cases:
        filepath = os.path.join(ROOT, "vision_engine/sample_tiles", filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error loading {filepath}")
            continue

        print(f"\n▶ Inspecting {tile_id} ({lat:.4f}, {lon:.4f}) | Initial Tabular Score: {initial_score:.2f}")

        # Run Perception-Decision-Action cycle
        decision, vision = engine.run_perception_decision_cycle(
            bgr_image=img,
            tile_id=tile_id,
            initial_tabular_score=initial_score,
            latitude=lat,
            longitude=lon,
        )

        # Save annotated OpenCV 5 image
        out_image_name = f"inspected_{tile_id}.png"
        out_image_path = os.path.join(output_dir, out_image_name)
        cv2.imwrite(out_image_path, vision.annotated_image)

        # Print Trace
        print(f"  • OpenCV 5 Dominant Class: {vision.dominant_class}")
        print(f"  • Canopy: {vision.canopy_fraction*100:.1f}% | Bare Ground: {vision.bare_ground_fraction*100:.1f}% | Impervious: {vision.impervious_fraction*100:.1f}%")
        print(f"  • Decision Branch: {decision.decision_branch}")
        print(f"  • Score Adjustment: {decision.initial_tabular_score:.2f} ──► {decision.adjusted_score:.2f}")
        print(f"  • Recommended Action: {decision.recommended_action}")
        print(f"  • Tools Dispatched: {', '.join(decision.tool_calls_invoked)}")
        print(f"  • Saved Visual Inspection Overlay: {out_image_path}")

        all_traces.append({
            "tile_id": decision.tile_id,
            "timestamp": decision.timestamp,
            "decision_branch": decision.decision_branch,
            "initial_score": decision.initial_tabular_score,
            "adjusted_score": decision.adjusted_score,
            "recommended_action": decision.recommended_action,
            "tools_invoked": decision.tool_calls_invoked,
            "perception_metrics": decision.perception_metrics,
            "reasoning_trace": decision.reasoning_trace,
            "action_payload": decision.action_payload,
            "inspection_image": out_image_path,
        })

    # Save complete JSON trace log
    json_path = os.path.join(output_dir, "agentic_decision_traces.json")
    with open(json_path, "w") as f:
        json.dump(all_traces, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Complete Agentic Decision Trace successfully saved to: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
