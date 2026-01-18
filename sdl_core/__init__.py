"""
Autonomous Drug Discovery Lab - Core Module

Self-driving laboratory framework for pharmaceutical research
combining Bayesian optimization with robotic automation.

Author: Oluwaseun O. Ajayi
Email: seunolanikeajayi@gmail.com
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Oluwaseun O. Ajayi"
__license__ = "MIT"
__email__ = "seunolanikeajayi@gmail.com"

from .orchestrator import SDLOrchestrator, OptimizationConfig, Experiment
from .experiment_designer import ExperimentDesigner, DesignSpace, AcquisitionFunction

__all__ = [
    'SDLOrchestrator',
    'OptimizationConfig',
    'Experiment',
    'ExperimentDesigner',
    'DesignSpace',
    'AcquisitionFunction',
    '__version__',
]