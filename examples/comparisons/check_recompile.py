import io
import re
import logging
from contextlib import contextmanager

import numpy as np
from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions
import jax
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_log_compiles", True)  # log compilation times for debugging  



@contextmanager
def capture_compile_log():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("jax._src.interpreters.pxla")

    old_level = logger.level
    old_handlers = list(logger.handlers)
    old_propagate = logger.propagate

    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.handlers = [handler]
    try:
        yield stream
    finally:
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        logger.handlers = old_handlers


# some input values for temperature (in K) and pressure (in bar)
T = 3000 #K
Nlayer = 100
temperature = np.full(Nlayer, T)
pressure = np.logspace(-8, 2, num=Nlayer)


# ExoGibbs comparison###############################################################
# Thermodynamic conditions
# from exogibbs.presets.ykb4 import chemsetup
from exogibbs.presets.fastchem import chemsetup

from exojax.utils.zsol import nsol
import jax.numpy as jnp

chem = chemsetup(path="fastchem/logK/logK_extended.dat", species_defalt_elements=False, element_file="fastchem/element_abundances/asplund_2020_extended.dat")
solar_abundance = nsol()
na_value = 1.e-14 # abundance for elements solar abundance is unavailable
nsol_vector = []
for el in chem.elements[:-1]:
    try:
        nsol_vector.append(solar_abundance[el])
    except:
        nsol_vector.append(na_value)
        print("no info on " ,el, "solar abundance. set",na_value)
nsol_vector = jnp.array([nsol_vector])  # no solar abundance for e-
element_vector = jnp.append(nsol_vector, 0.0)

import time
ts = time.time()
opts = EquilibriumOptions(method="scan_hot_from_top", epsilon_crit=1e-10, max_iter=1000) #1.07sec/run"
#opts = EquilibriumOptions(method="scan_hot_from_bottom", epsilon_crit=1e-10, max_iter=1000) #1.05sec/run
#opts = EquilibriumOptions(method="vmap_cold", epsilon_crit=1e-10, max_iter=1000) #2.11sec/run
niter = 5
temperature = temperature - niter
times = []

with capture_compile_log() as log:
    tset = time.time()
    for j in range(0, niter):
        temperature = temperature + 1.0
        res, diag = equilibrium_profile(
            chem,
            temperature,
            pressure,
            element_vector,
            Pref=1.0,
            options=opts,
            return_diagnostics=True
        )
        nk_result = res.x
        jax.block_until_ready(nk_result)
        print(diag["n_iter"][0], np.sum(diag["n_iter"][1:]))
        times.append(time.time() - tset)
        tset = time.time()

text = log.getvalue()
print("**************** diagnostics ********************")
print("scan_compile_count =", len(re.findall(r"Compiling scan\b", text)))
print("while_compile_count =", len(re.findall(r"Compiling while\b", text)))

te = time.time() - ts
print("times =", times)
print("mean_time (excl. 1st) =", sum(times[1:]) / len(times[1:]))
print("ExoGibbs calculation time:", te, "seconds")
##################################################################################
    
