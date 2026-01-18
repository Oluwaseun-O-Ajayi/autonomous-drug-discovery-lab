"""
Experiment Designer Module
Advanced experimental design using Bayesian optimization and active learning

Author: Oluwaseun O. Ajayi
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


@dataclass
class DesignSpace:
    """Defines the experimental design space"""
    continuous_params: Dict[str, Tuple[float, float]]
    categorical_params: Dict[str, List[str]] = None
    constraints: List[Callable] = None
    
    def get_dimensionality(self) -> int:
        """Return total dimensionality of design space"""
        return len(self.continuous_params)
    
    def validate_point(self, point: Dict) -> bool:
        """Check if experimental point satisfies all constraints"""
        if self.constraints is None:
            return True
        
        for constraint in self.constraints:
            if not constraint(point):
                return False
        return True


class AcquisitionFunction:
    """Acquisition functions for Bayesian optimization"""
    
    @staticmethod
    def expected_improvement(mu: np.ndarray, sigma: np.ndarray, 
                            best_f: float, xi: float = 0.01,
                            maximize: bool = True) -> np.ndarray:
        """
        Expected Improvement acquisition function
        
        Args:
            mu: Predicted mean from GP
            sigma: Predicted std from GP
            best_f: Current best objective value
            xi: Exploration parameter
            maximize: Whether to maximize (True) or minimize (False)
        """
        if maximize:
            improvement = mu - best_f - xi
        else:
            improvement = best_f - mu - xi
        
        Z = improvement / (sigma + 1e-9)
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        
        return ei
    
    @staticmethod
    def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray,
                              beta: float = 2.0,
                              maximize: bool = True) -> np.ndarray:
        """
        Upper Confidence Bound (UCB) acquisition function
        
        Args:
            mu: Predicted mean from GP
            sigma: Predicted std from GP
            beta: Exploration parameter (typically 1-3)
            maximize: Whether to maximize
        """
        if maximize:
            return mu + beta * sigma
        else:
            return -(mu - beta * sigma)
    
    @staticmethod
    def probability_of_improvement(mu: np.ndarray, sigma: np.ndarray,
                                  best_f: float, xi: float = 0.01,
                                  maximize: bool = True) -> np.ndarray:
        """
        Probability of Improvement (PI) acquisition function
        
        Args:
            mu: Predicted mean from GP
            sigma: Predicted std from GP
            best_f: Current best objective value
            xi: Exploration parameter
            maximize: Whether to maximize
        """
        if maximize:
            Z = (mu - best_f - xi) / (sigma + 1e-9)
        else:
            Z = (best_f - mu - xi) / (sigma + 1e-9)
        
        return norm.cdf(Z)


class ExperimentDesigner:
    """
    Advanced experiment designer using Bayesian optimization
    
    Designs optimal experiments based on Gaussian Process models
    and acquisition function optimization.
    """
    
    def __init__(self, 
                 design_space: DesignSpace,
                 acquisition_type: str = "EI",
                 kernel_type: str = "RBF",
                 n_restarts: int = 25):
        """
        Initialize experiment designer
        
        Args:
            design_space: Experimental design space
            acquisition_type: "EI", "UCB", or "PI"
            kernel_type: "RBF" or "Matern"
            n_restarts: Number of restarts for acquisition optimization
        """
        self.design_space = design_space
        self.acquisition_type = acquisition_type
        self.n_restarts = n_restarts
        
        # Initialize GP with appropriate kernel
        if kernel_type == "RBF":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0, 
                length_scale_bounds=(1e-2, 1e2)
            )
        elif kernel_type == "Matern":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        self.X_data = []
        self.y_data = []
        
        logger.info(f"Experiment designer initialized with {acquisition_type} acquisition")
    
    def propose_initial_designs(self, n_designs: int = 10,
                               method: str = "LHS") -> List[Dict[str, float]]:
        """
        Generate initial experimental designs
        
        Args:
            n_designs: Number of initial designs
            method: "LHS" (Latin Hypercube), "random", or "factorial"
        
        Returns:
            List of experimental parameter dictionaries
        """
        param_names = list(self.design_space.continuous_params.keys())
        n_params = len(param_names)
        
        if method == "LHS":
            # Latin Hypercube Sampling
            designs_normalized = self._latin_hypercube_sampling(n_designs, n_params)
        elif method == "random":
            # Random sampling
            designs_normalized = np.random.uniform(0, 1, (n_designs, n_params))
        elif method == "factorial":
            # Factorial design
            designs_normalized = self._factorial_design(n_params)
            n_designs = len(designs_normalized)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Convert to parameter dictionaries
        experiments = []
        for i in range(n_designs):
            exp_params = {}
            for j, param_name in enumerate(param_names):
                lower, upper = self.design_space.continuous_params[param_name]
                value = lower + designs_normalized[i, j] * (upper - lower)
                exp_params[param_name] = float(value)
            
            # Check constraints
            if self.design_space.validate_point(exp_params):
                experiments.append(exp_params)
        
        logger.info(f"Generated {len(experiments)} initial designs using {method}")
        return experiments
    
    def _latin_hypercube_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        """Generate Latin Hypercube samples"""
        samples = np.zeros((n_samples, n_dims))
        
        for i in range(n_dims):
            samples[:, i] = (np.random.permutation(n_samples) + 
                           np.random.uniform(0, 1, n_samples)) / n_samples
        
        return samples
    
    def _factorial_design(self, n_params: int, levels: int = 3) -> np.ndarray:
        """Generate factorial design"""
        # Create grid points
        grid_1d = np.linspace(0.1, 0.9, levels)
        
        # Generate all combinations
        grids = np.meshgrid(*[grid_1d] * n_params)
        designs = np.stack([g.flatten() for g in grids], axis=1)
        
        return designs
    
    def update_model(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update Gaussian Process model with new data
        
        Args:
            X_new: New experimental parameters (n_samples, n_features)
            y_new: New experimental results (n_samples,)
        """
        # Add to dataset
        if len(self.X_data) == 0:
            self.X_data = X_new.copy()
            self.y_data = y_new.copy()
        else:
            self.X_data = np.vstack([self.X_data, X_new])
            self.y_data = np.concatenate([self.y_data, y_new])
        
        # Fit GP
        self.gp.fit(self.X_data, self.y_data)
        
        logger.info(f"GP model updated with {len(X_new)} new experiments")
        logger.info(f"Total dataset size: {len(self.X_data)} experiments")
    
    def propose_next_experiment(self, maximize: bool = True,
                               exploration_weight: float = 0.01) -> Dict[str, float]:
        """
        Propose next experiment using acquisition function
        
        Args:
            maximize: Whether to maximize objective
            exploration_weight: Exploration parameter (xi for EI/PI, beta for UCB)
        
        Returns:
            Dictionary of proposed parameters
        """
        if len(self.X_data) == 0:
            raise ValueError("Must update model with data before proposing experiments")
        
        param_names = list(self.design_space.continuous_params.keys())
        bounds = [self.design_space.continuous_params[p] for p in param_names]
        
        # Get current best
        if maximize:
            best_f = np.max(self.y_data)
        else:
            best_f = np.min(self.y_data)
        
        # Define acquisition function to optimize
        def acquisition_wrapper(x):
            x_2d = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x_2d, return_std=True)
            
            if self.acquisition_type == "EI":
                acq = AcquisitionFunction.expected_improvement(
                    mu, sigma, best_f, exploration_weight, maximize
                )
            elif self.acquisition_type == "UCB":
                acq = AcquisitionFunction.upper_confidence_bound(
                    mu, sigma, exploration_weight, maximize
                )
            elif self.acquisition_type == "PI":
                acq = AcquisitionFunction.probability_of_improvement(
                    mu, sigma, best_f, exploration_weight, maximize
                )
            else:
                raise ValueError(f"Unknown acquisition type: {self.acquisition_type}")
            
            return -acq[0]  # Negative because we minimize
        
        # Optimize acquisition function with multiple restarts
        best_acq = np.inf
        best_x = None
        
        for _ in range(self.n_restarts):
            # Random starting point
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            
            # Optimize
            result = minimize(
                acquisition_wrapper,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        # Convert to parameter dictionary
        next_params = {name: float(val) for name, val in zip(param_names, best_x)}
        
        # Validate constraints
        if not self.design_space.validate_point(next_params):
            logger.warning("Proposed experiment violates constraints, using fallback")
            next_params = self._fallback_proposal(bounds)
        
        logger.info(f"Proposed next experiment with acquisition value: {-best_acq:.4f}")
        
        return next_params
    
    def _fallback_proposal(self, bounds: List[Tuple[float, float]]) -> Dict[str, float]:
        """Fallback to random sampling if optimization fails"""
        param_names = list(self.design_space.continuous_params.keys())
        return {
            name: float(np.random.uniform(b[0], b[1])) 
            for name, b in zip(param_names, bounds)
        }
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance from GP length scales
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if not hasattr(self.gp.kernel_, 'k2'):
            return {}
        
        try:
            length_scales = self.gp.kernel_.k2.length_scale
            if np.isscalar(length_scales):
                length_scales = np.array([length_scales])
            
            # Inverse of length scale indicates importance
            importance = 1.0 / np.array(length_scales)
            importance = importance / np.sum(importance)  # Normalize
            
            param_names = list(self.design_space.continuous_params.keys())
            return {name: float(imp) for name, imp in zip(param_names, importance)}
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}
    
    def predict_objective(self, X: np.ndarray, 
                         return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict objective value at given points
        
        Args:
            X: Points to predict (n_samples, n_features)
            return_std: Whether to return standard deviation
        
        Returns:
            Predicted means and (optionally) standard deviations
        """
        if return_std:
            mu, sigma = self.gp.predict(X, return_std=True)
            return mu, sigma
        else:
            mu = self.gp.predict(X, return_std=False)
            return mu, None
    
    def get_acquisition_surface(self, n_points: int = 50,
                               param1: str = None, param2: str = None,
                               fixed_params: Dict = None) -> np.ndarray:
        """
        Calculate acquisition function surface for visualization
        
        Args:
            n_points: Grid resolution
            param1, param2: Parameters to vary (first two if None)
            fixed_params: Fixed values for other parameters
        
        Returns:
            Grid of acquisition values
        """
        param_names = list(self.design_space.continuous_params.keys())
        
        if param1 is None:
            param1 = param_names[0]
        if param2 is None:
            param2 = param_names[1] if len(param_names) > 1 else param_names[0]
        
        # Create grid
        b1 = self.design_space.continuous_params[param1]
        b2 = self.design_space.continuous_params[param2]
        
        x1 = np.linspace(b1[0], b1[1], n_points)
        x2 = np.linspace(b2[0], b2[1], n_points)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Calculate acquisition for each grid point
        acq_grid = np.zeros_like(X1)
        
        for i in range(n_points):
            for j in range(n_points):
                params = fixed_params.copy() if fixed_params else {}
                params[param1] = X1[i, j]
                params[param2] = X2[i, j]
                
                # Convert to array
                x = np.array([params[p] for p in param_names]).reshape(1, -1)
                
                mu, sigma = self.gp.predict(x, return_std=True)
                best_f = np.max(self.y_data)
                
                acq = AcquisitionFunction.expected_improvement(
                    mu, sigma, best_f, 0.01, True
                )
                acq_grid[i, j] = acq[0]
        
        return acq_grid