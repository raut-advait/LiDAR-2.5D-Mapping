"""
Perception module adapter for Adaptive2_5DConverter.
Re-exports Adaptive2_5DConverter from main project's lidar_inference_pipeline.conversion.
"""
from lidar_inference_pipeline.conversion.adaptive_2_5d_converter import (
    Adaptive2_5DConverter,
)

__all__ = ["Adaptive2_5DConverter"]
