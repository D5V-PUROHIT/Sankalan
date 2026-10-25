"""
================================================================================
SANKALAN AI PLATFORM - GENERATION MODULE
File: backend/analytics/generation.py
Notice: The main Synthetic Data Generation implementation is located in:
        backend/analytics/synthetic_generation.py
================================================================================
This file re-exports all components from synthetic_generation.py for seamless compatibility.
"""

from .synthetic_generation import generate_and_inverse_transform

__all__ = ["generate_and_inverse_transform"]
