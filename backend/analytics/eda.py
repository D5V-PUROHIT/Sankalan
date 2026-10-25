"""
================================================================================
SANKALAN AI PLATFORM - EDA MODULE
File: backend/analytics/eda.py
Notice: The main Exploratory Data Analysis implementation is located in:
        backend/analytics/eda_analysis.py
================================================================================
This file re-exports all components from eda_analysis.py for seamless compatibility.
"""

from .eda_analysis import (
    is_valid_categorical,
    generate_eda_data,
    generate_comparison_data,
)

__all__ = [
    "is_valid_categorical",
    "generate_eda_data",
    "generate_comparison_data",
]
