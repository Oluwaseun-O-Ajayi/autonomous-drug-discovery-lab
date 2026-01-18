#!/usr/bin/env python3
"""
Complete Example: Autonomous Enzyme Optimization
Demonstrates end-to-end SDL workflow for enzymatic reaction optimization

Author: Oluwaseun O. Ajayi
License: MIT

Run: python examples/enzyme_optimization_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdl_core.orchestrator import SDLOrchestrator, OptimizationConfig


def simulate_enzyme_kinetics(temperature: float, pH: float, 
                            substrate_conc: float, enzyme_conc: float = 1.0) -> dict:
    """
    Simulate enzymatic reaction with realistic kinetics
    
    Models temperature and pH effects on enzyme activity,
    Michaelis-Menten kinetics, and experimental noise.
    
    Args:
        temperature: Reaction temperature (°C)
        pH: Reaction pH
        substrate_conc: Substrate concentration (μM)
        enzyme_conc: Enzyme concentration (μg/mL)
    
    Returns:
        Dictionary with experimental measurements
    """
    # Temperature dependence (Arrhenius with denaturation)
    # Optimal around 37°C
    if temperature < 45:
        temp_factor = np.exp(-((temperature - 37) / 12) ** 2)
    else:
        # Rapid denaturation above 45°C
        temp_factor = np.exp(-((temperature - 37) / 8) ** 2) * np.exp(-(temperature - 45) / 5)
    
    # pH dependence (bell-shaped curve)
    # Optimal around pH 7.4
    ph_factor = np.exp(-((pH - 7.4) / 1.2) ** 2)
    
    # Substrate inhibition at high concentrations
    if substrate_conc > 150:
        substrate_inhibition = np.exp(-(substrate_conc - 150) / 50)
    else:
        substrate_inhibition = 1.0
    
    # Michaelis-Menten kinetics
    Km = 45.0  # μM
    Vmax = 120.0 * temp_factor * ph_factor * substrate_inhibition * enzyme_conc
    velocity = (Vmax * substrate_conc) / (Km + substrate_conc)
    
    # Add realistic experimental noise (5% CV)
    noise = np.random.normal(0, 0.05 * velocity)
    observed_velocity = max(0, velocity + noise)
    
    # Simulate additional measurements
    absorbance = observed_velocity / 100  # Normalized
    
    # Quality control metrics
    temperature_stability = abs(np.random.normal(0, 0.1))
    pH_drift = abs(np.random.normal(0, 0.05))
    
    return {
        'reaction_velocity': observed_velocity,
        'absorbance_405nm': absorbance,
        'temperature_actual': temperature + np.random.normal(0, 0.1),
        'pH_actual': pH + np.random.normal(0, 0.05),
        'temperature_stability': temperature_stability,
        'pH_drift': pH_drift,
        'enzyme_stability': temp_factor * ph_factor,
        'time_seconds': 300
    }


def execute_experiment(params: dict) -> dict:
    """
    Execute enzyme assay experiment
    
    In production, this would interface with:
    - Liquid handling robots
    - Plate readers
    - Temperature controllers
    - pH meters
    """
    print(f"\n  Executing experiment:")
    print(f"    Temperature: {params['temperature']:.1f}°C")
    print(f"    pH: {params['pH']:.2f}")
    print(f"    Substrate: {params['substrate_conc']:.1f} μM")
    print(f"    Enzyme: {params['enzyme_conc']:.2f} μg/mL")
    
    # Simulate experiment
    results = simulate_enzyme_kinetics(
        temperature=params['temperature'],
        pH=params['pH'],
        substrate_conc=params['substrate_conc'],
        enzyme_conc=params['enzyme_conc']
    )
    
    print(f"    ✓ Velocity: {results['reaction_velocity']:.2f} μM/min")
    
    return results


def analyze_results(raw_data: dict) -> dict:
    """
    Analyze experimental results and perform quality control
    
    In production, this would:
    - Apply calibration curves
    - Calculate kinetic parameters
    - Perform statistical QC
    - Flag outliers
    """
    velocity = raw_data['reaction_velocity']
    
    # Quality control checks
    qc_checks = {
        'temperature_stable': raw_data['temperature_stability'] < 0.5,
        'pH_stable': raw_data['pH_drift'] < 0.2,
        'signal_adequate': raw_data['absorbance_405nm'] > 0.05,
        'not_saturated': raw_data['absorbance_405nm'] < 2.0,
    }
    
    qc_passed = all(qc_checks.values())
    
    # Calculate metrics
    specific_activity = velocity / raw_data.get('enzyme_conc', 1.0)
    
    return {
        'objective_value': velocity,  # What SDL optimizes
        'specific_activity': specific_activity,
        'qc_passed': qc_passed,
        'qc_details': qc_checks,
        'absorbance': raw_data['absorbance_405nm'],
        'enzyme_stability_index': raw_data['enzyme_stability']
    }


def visualize_results(sdl: SDLOrchestrator, results: dict, output_dir: Path):
    """Create comprehensive visualization of optimization results"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Get data
    iterations = results['optimization_trajectory']['iterations']
    obj_values = results['optimization_trajectory']['objective_values']
    cumulative_best = results['optimization_trajectory']['cumulative_best']
    X_data = np.array(sdl.X_data)
    
    # Plot 1: Optimization Trajectory
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(iterations, obj_values, 'o-', alpha=0.6, 
             markersize=6, label='Observed', color='steelblue')
    ax1.plot(iterations, cumulative_best, 'r-', linewidth=2.5, 
             label='Best So Far', color='crimson')
    ax1.axhline(y=results['best_result']['objective_value'], 
                color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label='Final Optimum')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reaction Velocity (μM/min)', fontsize=12, fontweight='bold')
    ax1.set_title('Optimization Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter Importance
    ax2 = fig.add_subplot(gs[0, 2])
    if results['parameter_importance']:
        params = list(results['parameter_importance'].keys())
        importance = list(results['parameter_importance'].values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
        bars = ax2.barh(params, importance, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax2.set_title('Parameter Importance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3-6: Parameter Evolution
    param_names = ['temperature', 'pH', 'substrate_conc', 'enzyme_conc']
    param_labels = ['Temperature (°C)', 'pH', 'Substrate (μM)', 'Enzyme (μg/mL)']
    best_params = results['best_result']['parameters']
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])
        param_idx = list(results['best_result']['parameters'].keys()).index(param)
        
        scatter = ax.scatter(iterations, X_data[:, param_idx], 
                           c=obj_values, cmap='plasma', 
                           s=100, edgecolor='black', linewidth=0.5, 
                           alpha=0.8)
        ax.axhline(y=best_params[param], color='red', 
                  linestyle='--', linewidth=2, label='Optimal', alpha=0.8)
        ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Evolution', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        if idx == 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Velocity (μM/min)', fontsize=10)
    
    # Plot 7: Performance Metrics
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Calculate metrics
    n_exp = results['campaign_summary']['completed_experiments']
    grid_search_exp = 10 ** 4  # 10 levels per parameter, 4 parameters
    efficiency = (grid_search_exp - n_exp) / grid_search_exp * 100
    time_saved = (grid_search_exp - n_exp) * 30 / 60  # hours
    
    metrics_text = f"""
    PERFORMANCE METRICS
    
    Experiments: {n_exp}
    vs Grid Search: {grid_search_exp}
    
    Efficiency: {efficiency:.1f}%
    Experiments Saved: {grid_search_exp - n_exp}
    
    Time Saved: {time_saved:.0f} hours
                ({time_saved/24:.1f} days)
    
    Optimal Velocity:
    {results['best_result']['objective_value']:.2f} μM/min
    """
    
    ax7.text(0.1, 0.5, metrics_text, fontsize=11, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax7.axis('off')
    
    # Overall title
    fig.suptitle('Autonomous Enzyme Optimization Results', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / 'optimization_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to {output_path}")
    
    plt.show()


def main():
    """Main execution function"""
    
    print("="*70)
    print("AUTONOMOUS DRUG DISCOVERY LAB")
    print("Enzyme Optimization Example")
    print("="*70)
    
    # Configure optimization
    config = OptimizationConfig(
        objective="maximize",
        parameter_space={
            'temperature': (25.0, 45.0),
            'pH': (6.0, 8.5),
            'substrate_conc': (20.0, 200.0),
            'enzyme_conc': (0.5, 3.0),
        },
        n_initial_experiments=12,
        max_iterations=35,
        convergence_threshold=0.01,
        exploration_weight=0.1
    )
    
    print("\nOptimization Configuration:")
    print(f"  Objective: {config.objective}")
    print(f"  Parameter space:")
    for param, (lower, upper) in config.parameter_space.items():
        print(f"    {param}: [{lower}, {upper}]")
    print(f"  Initial experiments: {config.n_initial_experiments}")
    print(f"  Max iterations: {config.max_iterations}")
    
    # Initialize SDL
    sdl = SDLOrchestrator(
        config=config,
        experiment_executor=execute_experiment,
        result_analyzer=analyze_results,
        output_dir="enzyme_optimization_output"
    )
    
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION CAMPAIGN")
    print("="*70)
    
    # Run optimization
    results = sdl.run_optimization_campaign()
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    best = results['best_result']
    print(f"\n✓ OPTIMAL CONDITIONS IDENTIFIED:")
    print(f"  Temperature:   {best['parameters']['temperature']:.2f} °C")
    print(f"  pH:            {best['parameters']['pH']:.2f}")
    print(f"  Substrate:     {best['parameters']['substrate_conc']:.1f} μM")
    print(f"  Enzyme:        {best['parameters']['enzyme_conc']:.2f} μg/mL")
    print(f"\n✓ OPTIMAL VELOCITY: {best['objective_value']:.2f} μM/min")
    
    summary = results['campaign_summary']
    print(f"\n✓ CAMPAIGN STATISTICS:")
    print(f"  Total experiments:     {summary['total_experiments']}")
    print(f"  Completed:             {summary['completed_experiments']}")
    print(f"  Failed:                {summary['failed_experiments']}")
    print(f"  Iterations:            {summary['iterations']}")
    
    if results['parameter_importance']:
        print(f"\n✓ PARAMETER IMPORTANCE:")
        for param, imp in results['parameter_importance'].items():
            print(f"  {param:20s}: {imp:.3f}")
    
    # Export data
    print(f"\n✓ Exporting data...")
    df = sdl.export_data("optimization_data.csv")
    print(f"  Data saved to: {sdl.output_dir / 'optimization_data.csv'}")
    
    # Visualize
    print(f"\n✓ Creating visualizations...")
    visualize_results(sdl, results, Path("enzyme_optimization_output"))
    
    # Calculate efficiency metrics
    n_exp = summary['completed_experiments']
    grid_exp = 10 ** 4
    efficiency = (grid_exp - n_exp) / grid_exp * 100
    time_saved = (grid_exp - n_exp) * 30 / 60 / 24  # days
    
    print(f"\n{'='*70}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'='*70}")
    print(f"  SDL experiments:       {n_exp}")
    print(f"  Grid search would need: {grid_exp:,}")
    print(f"  Efficiency gain:       {efficiency:.1f}%")
    print(f"  Experiments saved:     {grid_exp - n_exp:,}")
    print(f"  Time saved:            {time_saved:.1f} days")
    print(f"  Cost savings:          ${(grid_exp - n_exp) * 50:,.0f}")
    
    print(f"\n{'='*70}")
    print("READY FOR PUBLICATION")
    print(f"{'='*70}")
    print("All results, data, and figures are publication-ready.")
    print("Next steps:")
    print("  1. Validate optimal conditions experimentally (n=6)")
    print("  2. Include in Current Protocols manuscript")
    print("  3. Add to GitHub repository as case study")
    print(f"\nOutput directory: {sdl.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()