"""
generate_sample_tiles.py - Generates Optical Aerial Test Imagery for Delhi Micro-Zones
======================================================================================
Produces realistic synthetic optical tiles for key Delhi urban typology zones
to demonstrate the OpenCV 5 computer vision and agentic loop without external API keys.
"""

import os
from typing import Dict
import cv2
import numpy as np


def generate_sample_tiles(output_dir: str = "vision_engine/sample_tiles") -> Dict[str, str]:
    """Generate representative aerial tiles for Delhi micro-zones."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    h, w = 500, 500  # 500x500 pixels representing 250m x 250m tile (0.5m/px)

    # ── 1. Delhi Dense Urban (Connaught Place / Chandni Chowk matrix) ─────────
    # Dominated by concrete roofs, asphalt roads, minimal greenery
    dense_urban = np.full((h, w, 3), (128, 128, 132), dtype=np.uint8)  # Asphalt base
    # Add concrete rooftops
    for y in range(30, 470, 70):
        for x in range(30, 470, 70):
            roof_color = (np.random.randint(160, 210), np.random.randint(160, 210), np.random.randint(165, 215))
            cv2.rectangle(dense_urban, (x, y), (x + 55, y + 55), roof_color, -1)
            cv2.rectangle(dense_urban, (x, y), (x + 55, y + 55), (80, 80, 85), 2)  # Edge
    # Sparse roadside bushes (few trees)
    cv2.circle(dense_urban, (250, 20), 12, (34, 139, 34), -1)
    cv2.circle(dense_urban, (400, 20), 10, (30, 120, 30), -1)
    # Add realistic atmospheric noise
    noise = np.random.normal(0, 8, dense_urban.shape).astype(np.int16)
    dense_urban = np.clip(dense_urban.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    p1 = os.path.join(output_dir, "delhi_dense_concrete.png")
    cv2.imwrite(p1, dense_urban)
    paths["dense_concrete"] = p1

    # ── 2. Delhi Vacant Opportunity (Okhla Industrial vacant lot) ────────────
    # Dominated by large contiguous unpaved bare ground / soil
    bare_opportunity = np.full((h, w, 3), (135, 135, 135), dtype=np.uint8)
    # Surrounding peripheral roads and warehouses
    cv2.rectangle(bare_opportunity, (0, 0), (w, 50), (90, 90, 95), -1)
    cv2.rectangle(bare_opportunity, (0, 450), (w, h), (90, 90, 95), -1)
    # Large central bare land parcel (brown / yellowish soil tones)
    soil_pts = np.array([[70, 70], [430, 80], [410, 420], [90, 410]], np.int32)
    cv2.fillPoly(bare_opportunity, [soil_pts], (75, 130, 185))  # Brownish-tan soil in BGR
    # Minor sparse grass patch
    cv2.circle(bare_opportunity, (120, 120), 22, (38, 142, 40), -1)
    noise = np.random.normal(0, 10, bare_opportunity.shape).astype(np.int16)
    bare_opportunity = np.clip(bare_opportunity.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    p2 = os.path.join(output_dir, "delhi_bare_opportunity.png")
    cv2.imwrite(p2, bare_opportunity)
    paths["bare_opportunity"] = p2

    # ── 3. Delhi Existing Ridge Canopy (Northern Ridge / Nehru Park) ──────────
    # Dominated by dense tree clusters and multi-tier vegetation
    canopy_tile = np.full((h, w, 3), (45, 110, 40), dtype=np.uint8)  # Forest green base
    # Add dense canopy circles with varying green textures
    np.random.seed(42)
    for _ in range(80):
        cx = np.random.randint(20, w - 20)
        cy = np.random.randint(20, h - 20)
        radius = np.random.randint(18, 45)
        color = (np.random.randint(25, 45), np.random.randint(120, 190), np.random.randint(30, 60))
        cv2.circle(canopy_tile, (cx, cy), radius, color, -1)
    # Small access pathway
    cv2.line(canopy_tile, (0, 250), (w, 270), (120, 125, 130), 8)
    noise = np.random.normal(0, 6, canopy_tile.shape).astype(np.int16)
    canopy_tile = np.clip(canopy_tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    p3 = os.path.join(output_dir, "delhi_existing_canopy.png")
    cv2.imwrite(p3, canopy_tile)
    paths["existing_canopy"] = p3

    # ── 4. Delhi Yamuna Riverbank / Wetland ────────────────────────────────────
    # Dominated by river water and riparian sediment
    water_tile = np.full((h, w, 3), (120, 135, 140), dtype=np.uint8)
    # Flowing blue river polygon
    river_pts = np.array([[0, 120], [w, 80], [w, 360], [0, 400]], np.int32)
    cv2.fillPoly(water_tile, [river_pts], (180, 130, 40))  # River water in BGR (deep blue/cyan)
    # Riparian wetland vegetation on riverbanks
    cv2.ellipse(water_tile, (100, 80), (80, 25), 0, 0, 360, (35, 140, 35), -1)
    cv2.ellipse(water_tile, (380, 390), (90, 30), 0, 0, 360, (30, 130, 30), -1)
    noise = np.random.normal(0, 7, water_tile.shape).astype(np.int16)
    water_tile = np.clip(water_tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    p4 = os.path.join(output_dir, "delhi_yamuna_river.png")
    cv2.imwrite(p4, water_tile)
    paths["yamuna_river"] = p4

    return paths


if __name__ == "__main__":
    generated = generate_sample_tiles()
    print("Generated sample tiles:")
    for k, v in generated.items():
        print(f"  • {k}: {v}")
