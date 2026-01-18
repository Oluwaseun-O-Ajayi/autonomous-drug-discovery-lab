# Quick Start Guide

Get started with the Autonomous Drug Discovery Lab in 10 minutes.

---

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/Oluwaseun-O-Ajayi/autonomous-drug-discovery-lab.git
cd autonomous-drug-discovery-lab

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Your First SDL Campaign (5 minutes)

### Step 1: Define Your Experimental System

```python
from sdl_core.orchestrator import SDLOrchestrator, OptimizationConfig

# Define what happens in each experiment
def run_experiment(params):
    """Your experiment execution code here"""
    # Interface with your instruments
    temperature = params['temperature']
    pH = params['pH']
    
    # Run experiment and collect data
    result = my_instrument.measure(temperature, pH)
    
    return result

# Define how to analyze results
def analyze_results(raw_data):
    """Your analysis code here"""
    return {
        'objective_value': calculate_activity(raw_data),
        'qc_passed': check_quality(raw_data)
    }
```

### Step 2: Configure Optimization

```python
# Define parameter space and optimization settings
config = OptimizationConfig(
    objective="maximize",  # or "minimize"
    parameter_space={
        'temperature': (20.0, 50.0),  # min, max
        'pH': (5.0, 9.0),
        'concentration': (10.0, 200.0),
    },
    n_initial_experiments=10,
    max_iterations=30
)
```

### Step 3: Run SDL Campaign

```python
# Initialize orchestrator
sdl = SDLOrchestrator(
    config=config,
    experiment_executor=run_experiment,
    result_analyzer=analyze_results,
    output_dir="my_optimization"
)

# Run autonomous optimization
results = sdl.run_optimization_campaign()

# Get optimal conditions
best_params = results['best_result']['parameters']
best_value = results['best_result']['objective_value']

print(f"Optimal conditions: {best_params}")
print(f"Optimal value: {best_value}")
```

### Step 4: Export and Visualize

```python
# Export all data to CSV
df = sdl.export_data("results.csv")

# Results automatically saved in output_dir/
# - campaign_report_TIMESTAMP.json
# - results.csv
# - optimization plots
```

---

## Example: Enzyme Optimization

Run the complete enzyme optimization example:

```bash
python examples/enzyme_optimization_example.py
```

This demonstrates:
- Simulated enzyme kinetics experiments
- 4-parameter optimization (temperature, pH, substrate, enzyme)
- ~40 autonomous experiments
- Publication-ready visualizations
- Complete data export

---

## Example: ADMET Screening

Screen compounds using ADMET predictions:

```python
from integrations.admet_predictor import ADMETPredictor, create_admet_screening_workflow

# List of compounds to screen
compounds = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    # ... more SMILES strings
]

# Create screening workflow
workflow = create_admet_screening_workflow(
    compounds=compounds,
    min_drugability=0.6
)

# Run screening
predictor = workflow['predictor']
results = predictor.batch_predict([{'smiles': s} for s in compounds])

# Filter promising compounds
promising = predictor.filter_compounds(results, min_score=0.7)
```

---

## Integration with Your Existing Tools

The SDL framework integrates with your existing automation toolkit:

```python
# Import your existing tools
from drugability_toolkit import ADMETCalculator
from lcms_data_processor import LCMSAnalyzer
from robot_workcell_simulator import RobotController

# Use in SDL workflow
def integrated_experiment(params):
    # Predict first
    admet = ADMETCalculator().predict(compound)
    
    if admet['drugability'] > 0.7:
        # Execute physical experiment
        robot = RobotController()
        robot.prepare_sample(params)
        
        result = LCMSAnalyzer().quantify(sample)
        return result
    
    return {'skip': True}
```

---

## Configuration Options

### Optimization Config

```python
OptimizationConfig(
    objective="maximize",           # or "minimize"
    parameter_space={...},          # dict of param: (min, max)
    acquisition_function="EI",      # "EI", "UCB", or "PI"
    n_initial_experiments=10,       # initial random sampling
    max_iterations=50,              # max optimization iterations
    convergence_threshold=0.01,     # stop if improvement < 1%
    exploration_weight=0.1,         # balance explore/exploit
)
```

### Acquisition Functions

- **EI (Expected Improvement)**: Balanced, good default
- **UCB (Upper Confidence Bound)**: More exploratory
- **PI (Probability of Improvement)**: More exploitative

### When to Use What

| Scenario | Acquisition | Initial Exps | Iterations |
|----------|-------------|--------------|------------|
| 2-3 parameters | EI | 8-12 | 20-30 |
| 4-5 parameters | EI | 15-20 | 30-50 |
| 6+ parameters | UCB | 25-40 | 50-100 |
| Rugged landscape | UCB | 15-25 | 40-80 |
| Smooth landscape | PI | 10-15 | 25-40 |

---

## Common Patterns

### Pattern 1: Multi-Stage Optimization

```python
# Stage 1: Coarse search
config_coarse = OptimizationConfig(
    parameter_space={'temp': (20, 50), 'pH': (5, 9)},
    n_initial_experiments=15,
    max_iterations=30
)
results_coarse = sdl.run_optimization_campaign()

# Stage 2: Fine search around optimum
best = results_coarse['best_result']['parameters']
config_fine = OptimizationConfig(
    parameter_space={
        'temp': (best['temp']-3, best['temp']+3),
        'pH': (best['pH']-0.5, best['pH']+0.5)
    },
    n_initial_experiments=8,
    max_iterations=20
)
results_fine = sdl.run_optimization_campaign()
```

### Pattern 2: Constraint Handling

```python
def constrained_experiment(params):
    # Check constraints before running
    if params['temperature'] > 45 and params['pH'] > 8:
        # Unstable combination
        return {'skip': True}
    
    # Run experiment
    return normal_experiment(params)
```

### Pattern 3: Multi-Objective Optimization

```python
def analyze_multi_objective(raw_data):
    activity = calculate_activity(raw_data)
    stability = calculate_stability(raw_data)
    
    # Weighted combination
    objective = 0.7 * activity + 0.3 * stability
    
    return {
        'objective_value': objective,
        'activity': activity,
        'stability': stability
    }
```

---

## Troubleshooting

### Issue: SDL proposes invalid parameters

**Solution**: Add validation in executor
```python
def safe_experiment(params):
    # Validate before executing
    if not validate_params(params):
        return {'skip': True, 'reason': 'invalid'}
    return experiment(params)
```

### Issue: Slow convergence

**Solutions**:
- Increase `n_initial_experiments`
- Switch to `UCB` acquisition
- Increase `exploration_weight`

### Issue: Stuck in local optimum

**Solutions**:
- Restart with wider parameter bounds
- Use `UCB` for more exploration
- Increase `exploration_weight` to 0.2-0.3

### Issue: Experiments failing frequently

**Solutions**:
- Add retry logic in executor
- Implement better error handling
- Log failures for debugging
- Reduce parameter space if instability region

---

## Next Steps

1. **Run Examples**: Try the provided examples to understand the workflow
2. **Adapt to Your System**: Replace simulation functions with your instruments
3. **Start Small**: Begin with 2-3 parameters, expand later
4. **Read Protocols**: Check `docs/protocols/` for detailed protocols
5. **Review Case Studies**: See `docs/case_studies/` for real applications

---

## Getting Help

- **Documentation**: Full docs in `docs/`
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Examples**: More examples in `examples/notebooks/`

---

## Citation

If you use this framework, please cite:

```bibtex
@software{ajayi2025autonomous,
  title={Autonomous Drug Discovery Lab},
  author={Ajayi, Oluwaseun O.},
  year={2025},
  url={https://github.com/Oluwaseun-O-Ajayi/autonomous-drug-discovery-lab}
}
```

---

**You're ready to start autonomous experimentation! 🚀**