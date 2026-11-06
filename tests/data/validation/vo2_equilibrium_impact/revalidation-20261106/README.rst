VO2 historical replay amendment
================================

This maintenance preserves the 2026-11-02 execution receipt, all numerical
outputs and the original source packets. Files under ``source/`` are exact
bytes from PR head 5978ab5294fffad861fc1663b089e09e8f804195, not executable
replacements for the current implementation.

The current helper names curated-c2h-v1 explicitly. The current offline test
compares all four model Gibbs vectors at all 75 archived conditions, resolves
the original selected hashes against historical bytes, and pins the current
helper/test in ``revalidation.json``. This amendment does not claim new
equilibrium solves, a changed source evaluation, or a production adoption.
