"""
================================================================================
SANKALAN AI PLATFORM - PREPROCESSING MODULE
File: backend/analytics/preprocessing.py
Notice: The main Data Preprocessing implementation is located in:
        backend/analytics/data_preprocessing.py
================================================================================
This file re-exports all components from data_preprocessing.py for seamless compatibility.
"""

from .data_preprocessing import clean_and_preprocess

__all__ = ["clean_and_preprocess"]
