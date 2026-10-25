"""
================================================================================
SANKALAN AI PLATFORM - EVALUATION MODULE
File: backend/analytics/evaluation.py
Notice: The main Evaluation & Quality Metrics implementation is located in:
        backend/analytics/synthetic_evaluation.py
================================================================================
This file re-exports all components from synthetic_evaluation.py for seamless compatibility.
"""

from .synthetic_evaluation import evaluate_synthetic_data

__all__ = ["evaluate_synthetic_data"]
