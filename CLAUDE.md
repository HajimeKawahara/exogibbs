# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExoGibbs is an auto-differentiable thermochemical equilibrium solver implemented in JAX. It computes chemical equilibrium states by minimizing Gibbs free energy, with applications in planetary atmosphere modeling and chemical process simulation.

## Key Architecture

- **Core equilibrium calculations**: `src/exogibbs/equilibrium/gibbs.py` - Contains Gibbs energy computation and chemical potential interpolation
- **Optimization algorithms**: `src/exogibbs/optimize/` - KL mirror descent (`klmirror.py`) and projected gradient descent (`naive_pgd.py`) for constrained optimization
- **Data handling**: `src/exogibbs/io/load_data.py` - Loads JANAF thermochemical data
- **Stoichiometry**: `src/exogibbs/stoichiometry/analyze_formula_matrix.py` - Chemical formula matrix analysis
- **Test data generation**: `src/exogibbs/test/generate_gibbs.py` - Creates test cases for equilibrium calculations

## Development Commands

**Installation**: 
```bash
pip install -e .
```

**Testing**:
```bash
python -m pytest tests/unittests/
```

**Key Dependencies**: JAX/JAXlib for auto-differentiation, pandas for data handling, tqdm for progress bars

## Important Notes

- The project uses JAX for auto-differentiation and JIT compilation - ensure JAX operations are used consistently
- Chemical potential data is interpolated from JANAF tables stored in `src/exogibbs/data/`
- Optimization operates in log-space to handle numerical stability with exponential functions
- Formula matrices encode stoichiometric constraints for mass balance during equilibrium calculation