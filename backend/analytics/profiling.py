"""
================================================================================
SANKALAN AI PLATFORM - DATASET PROFILING MODULE
File: backend/analytics/profiling.py
Notice: The main Dataset Profiling implementation is located in:
        backend/analytics/dataset_profiling.py
================================================================================
This file re-exports all components from dataset_profiling.py for seamless compatibility.
"""

from .dataset_profiling import profile_dataset

__all__ = ["profile_dataset"]
