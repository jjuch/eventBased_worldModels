"""Synthetic rigid-sphere trajectory generator.

Imports are intentionally lightweight so the mechanics can be tested without loading
optional storage backends. Import DatasetGenerator from ball_simulator.generator.
"""
from .config import ExperimentConfig
from .simulator import BallSimulator

__all__ = ["ExperimentConfig", "BallSimulator"]
