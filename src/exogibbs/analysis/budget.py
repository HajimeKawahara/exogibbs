from typing import Dict, Optional

import jax.numpy as jnp

from exogibbs.api.chemistry import ChemicalSetup

def element_budget(
    chem_gas: ChemicalSetup,
    ln_ngas: jnp.ndarray,
    chem_cond: Optional[ChemicalSetup] = None,
    ln_ncond: Optional[jnp.ndarray] = None,
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
    """computes element budget in molecules (gas species and condensates)

    Args:
        chem_gas: ChemicalSetup
            The chemical setup for gas phase.
        ln_ngas: jnp.ndarray
            Logarithm of amounts of gas species (K_gas,) or (N, K_gas).
        chem_cond: Optional[ChemicalSetup]
            The chemical setup for condensed phase.
        ln_ncond: Optional[jnp.ndarray]
            Logarithm of amounts of condensed species (K_cond,) or (N, K_cond).

    Returns:
        element budget
            such as eb["H"] = {"H2": values, "CH4": values, ...}
            sum(eb["H"].values()) should correspond to
            chem_gas.element_vector_reference[chem_gas.elements.index("H")]
        

    
    """
    ln_ngas = jnp.asarray(ln_ngas)
    if ln_ngas.ndim not in (1, 2):
        raise ValueError("ln_ngas must have shape (K_gas,) or (N, K_gas).")
    single_state = ln_ngas.ndim == 1
    ln_ngas_2d = ln_ngas[None, :] if single_state else ln_ngas

    formula_gas = jnp.asarray(chem_gas.formula_matrix)
    if formula_gas.ndim != 2:
        raise ValueError("chem_gas.formula_matrix must have shape (E, K_gas).")
    if formula_gas.shape[1] != ln_ngas_2d.shape[1]:
        raise ValueError(
            "chem_gas.formula_matrix and ln_ngas have incompatible shapes."
        )

    n_gas = jnp.exp(ln_ngas_2d)
    contrib_gas = n_gas[:, None, :] * formula_gas[None, :, :]

    element_count = formula_gas.shape[0]
    if chem_gas.elements is not None:
        if len(chem_gas.elements) != element_count:
            raise ValueError(
                "chem_gas.elements length must match chem_gas.formula_matrix rows."
            )
        element_names = tuple(chem_gas.elements)
    else:
        element_names = tuple(f"E{i}" for i in range(element_count))

    gas_species = (
        tuple(chem_gas.species)
        if chem_gas.species is not None
        else tuple(f"gas_{i}" for i in range(formula_gas.shape[1]))
    )

    cond_species = ()
    contrib_cond = None
    if (chem_cond is None) != (ln_ncond is None):
        raise ValueError("chem_cond and ln_ncond must both be provided or both be None.")

    if chem_cond is not None and ln_ncond is not None:
        ln_ncond = jnp.asarray(ln_ncond)
        if ln_ncond.ndim not in (1, 2):
            raise ValueError("ln_ncond must have shape (K_cond,) or (N, K_cond).")
        if (ln_ncond.ndim == 1) != single_state:
            raise ValueError(
                "ln_ngas and ln_ncond must both be 1D or both be 2D."
            )
        ln_ncond_2d = ln_ncond[None, :] if ln_ncond.ndim == 1 else ln_ncond

        formula_cond = jnp.asarray(chem_cond.formula_matrix)
        if formula_cond.ndim != 2:
            raise ValueError("chem_cond.formula_matrix must have shape (E, K_cond).")
        if formula_cond.shape[1] != ln_ncond_2d.shape[1]:
            raise ValueError(
                "chem_cond.formula_matrix and ln_ncond have incompatible shapes."
            )
        if ln_ncond_2d.shape[0] != ln_ngas_2d.shape[0]:
            raise ValueError(
                "ln_ngas and ln_ncond must have the same batch length in 2D mode."
            )

        if chem_cond.elements is not None and chem_gas.elements is not None:
            cond_pos = {e: i for i, e in enumerate(chem_cond.elements)}
            try:
                row_idx = jnp.asarray([cond_pos[e] for e in element_names], dtype=jnp.int32)
            except KeyError as exc:
                raise ValueError(
                    "chem_cond.elements must include all gas elements for alignment."
                ) from exc
            formula_cond = formula_cond[row_idx, :]
        elif formula_cond.shape[0] != element_count:
            raise ValueError(
                "chem_cond.formula_matrix must have same number of rows as gas matrix "
                "when element names are not available for alignment."
            )

        n_cond = jnp.exp(ln_ncond_2d)
        contrib_cond = n_cond[:, None, :] * formula_cond[None, :, :]
        cond_species = (
            tuple(chem_cond.species)
            if chem_cond.species is not None
            else tuple(f"cond_{i}" for i in range(formula_cond.shape[1]))
        )

    overlap = set(gas_species).intersection(cond_species)
    gas_labels = tuple(f"gas:{s}" if s in overlap else s for s in gas_species)
    cond_labels = tuple(f"cond:{s}" if s in overlap else s for s in cond_species)

    eb: Dict[str, Dict[str, jnp.ndarray]] = {}
    for e_idx, element_name in enumerate(element_names):
        species_budget = {
            species: contrib_gas[:, e_idx, s_idx]
            for s_idx, species in enumerate(gas_labels)
            if float(formula_gas[e_idx, s_idx]) != 0.0
        }
        if contrib_cond is not None:
            species_budget.update(
                {
                    species: contrib_cond[:, e_idx, s_idx]
                    for s_idx, species in enumerate(cond_labels)
                    if float(formula_cond[e_idx, s_idx]) != 0.0
                }
            )
        if single_state:
            species_budget = {
                name: values[0] for name, values in species_budget.items()
            }
        eb[element_name] = species_budget

    return eb
