"""
Autonomous Drug Discovery Lab - Core Orchestrator
Self-driving laboratory engine for closed-loop pharmaceutical research

Author: Oluwaseun O. Ajayi
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Represents a single experiment in the SDL workflow"""
    exp_id: str
    parameters: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, running, completed, failed
    results: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert experiment to dictionary for serialization"""
        return {
            'exp_id': self.exp_id,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'results': self.results,
            'metadata': self.metadata
        }


@dataclass
class OptimizationConfig:
    """Configuration for SDL optimization campaign"""
    objective: str  # maximize or minimize
    parameter_space: Dict[str, Tuple[float, float]]  # parameter bounds
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    acquisition_function: str = "EI"  # EI, UCB, PI
    n_initial_experiments: int = 5
    max_iterations: int = 50
    convergence_threshold: float = 0.001
    exploration_weight: float = 0.1


class ExperimentDesigner:
    """Bayesian optimization for experimental design"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gp = None
        self._initialize_gp()
        
    def _initialize_gp(self):
        """Initialize Gaussian Process with RBF kernel"""
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
    
    def propose_initial_experiments(self) -> List[Dict[str, float]]:
        """Generate initial experimental designs using Latin Hypercube sampling"""
        n_params = len(self.config.parameter_space)
        n_exp = self.config.n_initial_experiments
        
        # Latin Hypercube Sampling
        lhs = np.random.uniform(0, 1, (n_exp, n_params))
        
        experiments = []
        param_names = list(self.config.parameter_space.keys())
        
        for i in range(n_exp):
            exp_params = {}
            for j, param_name in enumerate(param_names):
                lower, upper = self.config.parameter_space[param_name]
                exp_params[param_name] = lower + lhs[i, j] * (upper - lower)
            experiments.append(exp_params)
        
        return experiments
    
    def update_model(self, X: np.ndarray, y: np.ndarray):
        """Update Gaussian Process with new experimental data"""
        self.gp.fit(X, y)
        logger.info(f"GP model updated with {len(X)} experiments")
    
    def _acquisition_function(self, X: np.ndarray, X_samples: np.ndarray, 
                             y_samples: np.ndarray) -> np.ndarray:
        """Calculate acquisition function (Expected Improvement)"""
        mu, sigma = self.gp.predict(X.reshape(-1, len(self.config.parameter_space)), 
                                    return_std=True)
        
        if self.config.objective == "maximize":
            best_f = np.max(y_samples)
            imp = mu - best_f - self.config.exploration_weight
        else:
            best_f = np.min(y_samples)
            imp = best_f - mu - self.config.exploration_weight
        
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        
        return ei
    
    def propose_next_experiment(self, X_samples: np.ndarray, 
                               y_samples: np.ndarray) -> Dict[str, float]:
        """Propose next experiment using Bayesian optimization"""
        # Define bounds for optimization
        param_names = list(self.config.parameter_space.keys())
        bounds = [self.config.parameter_space[p] for p in param_names]
        
        # Random restart optimization
        best_acq = -np.inf
        best_x = None
        
        for _ in range(25):
            # Random starting point
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            
            # Minimize negative acquisition function
            result = minimize(
                lambda x: -self._acquisition_function(x, X_samples, y_samples)[0],
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x
        
        # Convert to parameter dictionary
        next_params = {name: float(val) for name, val in zip(param_names, best_x)}
        
        return next_params


class SDLOrchestrator:
    """
    Main orchestrator for self-driving laboratory workflows
    
    Coordinates experiment design, execution, analysis, and optimization
    in a closed-loop autonomous fashion.
    """
    
    def __init__(self, 
                 config: OptimizationConfig,
                 experiment_executor: Callable,
                 result_analyzer: Callable,
                 output_dir: str = "sdl_output"):
        """
        Initialize SDL Orchestrator
        
        Args:
            config: Optimization configuration
            experiment_executor: Function to execute experiments (robot/instruments)
            result_analyzer: Function to analyze experimental results
            output_dir: Directory for output files and logs
        """
        self.config = config
        self.experiment_executor = experiment_executor
        self.result_analyzer = result_analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.designer = ExperimentDesigner(config)
        self.experiments: List[Experiment] = []
        self.iteration = 0
        
        # Storage for optimization data
        self.X_data = []
        self.y_data = []
        
        logger.info("SDL Orchestrator initialized")
    
    def run_optimization_campaign(self) -> Dict[str, Any]:
        """
        Execute complete SDL optimization campaign
        
        Returns:
            Dictionary with optimization results and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting SDL Optimization Campaign")
        logger.info("=" * 60)
        
        # Phase 1: Initial exploration
        logger.info("\n[Phase 1] Initial Exploration")
        initial_params = self.designer.propose_initial_experiments()
        
        for i, params in enumerate(initial_params):
            exp = self._execute_experiment(params, f"init_{i+1}")
            if exp.status == "completed":
                self._add_to_dataset(exp)
        
        # Update GP model with initial data
        if len(self.X_data) > 0:
            X = np.array(self.X_data)
            y = np.array(self.y_data)
            self.designer.update_model(X, y)
        
        # Phase 2: Iterative optimization
        logger.info("\n[Phase 2] Bayesian Optimization")
        converged = False
        prev_best = None
        
        while self.iteration < self.config.max_iterations and not converged:
            self.iteration += 1
            logger.info(f"\n--- Iteration {self.iteration} ---")
            
            # Propose next experiment
            X = np.array(self.X_data)
            y = np.array(self.y_data)
            next_params = self.designer.propose_next_experiment(X, y)
            
            # Execute experiment
            exp = self._execute_experiment(next_params, f"opt_{self.iteration}")
            
            if exp.status == "completed":
                self._add_to_dataset(exp)
                
                # Update model
                X = np.array(self.X_data)
                y = np.array(self.y_data)
                self.designer.update_model(X, y)
                
                # Check convergence
                current_best = np.max(y) if self.config.objective == "maximize" else np.min(y)
                if prev_best is not None:
                    improvement = abs(current_best - prev_best) / abs(prev_best)
                    if improvement < self.config.convergence_threshold:
                        logger.info(f"Converged! Improvement: {improvement:.6f}")
                        converged = True
                
                prev_best = current_best
                logger.info(f"Current best: {current_best:.4f}")
        
        # Generate final report
        results = self._generate_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("SDL Campaign Completed")
        logger.info("=" * 60)
        
        return results
    
    def _execute_experiment(self, params: Dict[str, float], exp_id: str) -> Experiment:
        """Execute a single experiment"""
        exp = Experiment(exp_id=exp_id, parameters=params, status="running")
        self.experiments.append(exp)
        
        logger.info(f"Executing experiment {exp_id}")
        logger.info(f"Parameters: {params}")
        
        try:
            # Execute experiment using provided executor function
            raw_results = self.experiment_executor(params)
            
            # Analyze results
            analyzed_results = self.result_analyzer(raw_results)
            
            exp.results = analyzed_results
            exp.status = "completed"
            
            logger.info(f"Experiment {exp_id} completed successfully")
            logger.info(f"Results: {analyzed_results}")
            
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {str(e)}")
            exp.status = "failed"
            exp.metadata['error'] = str(e)
        
        return exp
    
    def _add_to_dataset(self, exp: Experiment):
        """Add completed experiment to dataset"""
        if exp.results and 'objective_value' in exp.results:
            param_names = list(self.config.parameter_space.keys())
            X_point = [exp.parameters[p] for p in param_names]
            y_point = exp.results['objective_value']
            
            self.X_data.append(X_point)
            self.y_data.append(y_point)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        completed_exps = [e for e in self.experiments if e.status == "completed"]
        
        y_array = np.array(self.y_data)
        
        if self.config.objective == "maximize":
            best_idx = np.argmax(y_array)
            best_value = np.max(y_array)
        else:
            best_idx = np.argmin(y_array)
            best_value = np.min(y_array)
        
        best_exp = completed_exps[best_idx]
        
        report = {
            'campaign_summary': {
                'total_experiments': len(self.experiments),
                'completed_experiments': len(completed_exps),
                'failed_experiments': len([e for e in self.experiments if e.status == "failed"]),
                'iterations': self.iteration,
                'objective': self.config.objective,
            },
            'best_result': {
                'experiment_id': best_exp.exp_id,
                'parameters': best_exp.parameters,
                'objective_value': best_value,
                'all_results': best_exp.results
            },
            'optimization_trajectory': {
                'iterations': list(range(len(self.y_data))),
                'objective_values': self.y_data,
                'cumulative_best': self._get_cumulative_best(y_array)
            },
            'parameter_importance': self._calculate_parameter_importance()
        }
        
        # Save report
        report_path = self.output_dir / f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        
        return report
    
    def _get_cumulative_best(self, y_array: np.ndarray) -> List[float]:
        """Calculate cumulative best values"""
        if self.config.objective == "maximize":
            return np.maximum.accumulate(y_array).tolist()
        else:
            return np.minimum.accumulate(y_array).tolist()
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance using GP length scales"""
        if self.designer.gp is None or not hasattr(self.designer.gp.kernel_, 'k2'):
            return {}
        
        try:
            length_scales = self.designer.gp.kernel_.k2.length_scale
            if np.isscalar(length_scales):
                length_scales = [length_scales]
            
            # Inverse of length scale indicates importance
            importance = 1.0 / np.array(length_scales)
            importance = importance / np.sum(importance)  # Normalize
            
            param_names = list(self.config.parameter_space.keys())
            return {name: float(imp) for name, imp in zip(param_names, importance)}
        except:
            return {}
    
    def export_data(self, filename: str = "sdl_data.csv"):
        """Export experimental data to CSV"""
        import pandas as pd
        
        data_rows = []
        for exp in self.experiments:
            if exp.status == "completed" and exp.results:
                row = {**exp.parameters, **exp.results, 'exp_id': exp.exp_id}
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Data exported to {output_path}")
        
        return df


# Example usage and demonstration
if __name__ == "__main__":
    # Example: Optimize enzyme reaction conditions
    
    def mock_enzyme_experiment(params: Dict[str, float]) -> Dict[str, Any]:
        """Simulate enzyme kinetics experiment"""
        # Simulate Michaelis-Menten with optimal conditions
        temp = params['temperature']
        ph = params['pH']
        substrate = params['substrate_conc']
        
        # Optimal around 37°C, pH 7.4
        temp_factor = np.exp(-((temp - 37) / 10) ** 2)
        ph_factor = np.exp(-((ph - 7.4) / 1.5) ** 2)
        
        # Michaelis-Menten kinetics
        Km = 50  # μM
        Vmax = 100 * temp_factor * ph_factor
        velocity = (Vmax * substrate) / (Km + substrate)
        
        # Add noise
        velocity += np.random.normal(0, 2)
        
        return {
            'reaction_velocity': velocity,
            'activity': velocity / 100,  # Normalized
        }
    
    def analyze_enzyme_results(raw_results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze enzyme experiment results"""
        return {
            'objective_value': raw_results['reaction_velocity'],
            'normalized_activity': raw_results['activity']
        }
    
    # Configure optimization
    config = OptimizationConfig(
        objective="maximize",
        parameter_space={
            'temperature': (20.0, 50.0),  # °C
            'pH': (5.0, 9.0),
            'substrate_conc': (10.0, 200.0),  # μM
        },
        n_initial_experiments=8,
        max_iterations=25,
        convergence_threshold=0.005
    )
    
    # Initialize and run SDL
    sdl = SDLOrchestrator(
        config=config,
        experiment_executor=mock_enzyme_experiment,
        result_analyzer=analyze_enzyme_results,
        output_dir="example_sdl_output"
    )
    
    results = sdl.run_optimization_campaign()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest conditions found:")
    print(f"Temperature: {results['best_result']['parameters']['temperature']:.2f} °C")
    print(f"pH: {results['best_result']['parameters']['pH']:.2f}")
    print(f"Substrate: {results['best_result']['parameters']['substrate_conc']:.2f} μM")
    print(f"\nOptimal velocity: {results['best_result']['objective_value']:.2f}")
    print(f"Total experiments: {results['campaign_summary']['completed_experiments']}")
    
    # Export data
    sdl.export_data()