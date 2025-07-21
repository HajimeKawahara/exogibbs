# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExoGibbs is an auto-differentiable thermochemical equilibrium solver implemented in JAX. It computes chemical equilibrium states by minimizing Gibbs free energy, with applications in planetary atmosphere modeling and chemical process simulation.

## Key Architecture

- **Core equilibrium calculations**: `src/exogibbs/equilibrium/gibbs.py` - Contains Gibbs energy computation and chemical potential interpolation
- **Optimization algorithms**: `src/exogibbs/optimize/` - KL mirror descent (`klmirror.py`), projected gradient descent (`naive_pgd.py`), and Lagrange multiplier methods (`lagrange.py`) for constrained optimization
- **Data handling**: `src/exogibbs/io/load_data.py` - Loads JANAF thermochemical data from `src/exogibbs/data/`
- **Stoichiometry**: `src/exogibbs/stoichiometry/analyze_formula_matrix.py` - Chemical formula matrix analysis for mass balance constraints
- **Test data generation**: `src/exogibbs/test/generate_gibbs.py` - Creates test cases for equilibrium calculations
- **Testing**: Uses pytest framework with tests in `tests/unittests/` organized by module structure

## Development Commands

**Installation**: 
```bash
pip install -e .
```

**Testing**:
```bash
# Run all tests
python -m pytest tests/unittests/

# Run specific test file
python -m pytest tests/unittests/equilibrium/gibbs_test.py

# Run individual test function
python -m pytest tests/unittests/equilibrium/gibbs_test.py::test_total_gibbs_energy
```

**Key Dependencies**: JAX/JAXlib for auto-differentiation, pandas for data handling, pytest for testing

## Important Notes

- **JAX Compatibility**: The project uses JAX for auto-differentiation and JIT compilation - ensure JAX operations are used consistently. Be careful with JAX compatibility when adding new functions
- **Keep It Simple**: Stay simple, do not add complex tests and functions
- Chemical potential data is interpolated from JANAF tables stored in `src/exogibbs/data/`
- Optimization operates in log-space to handle numerical stability with exponential functions
- Formula matrices encode stoichiometric constraints for mass balance during equilibrium calculation