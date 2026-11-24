Historical condensate audit sources
====================================

The source tree preserves exact bytes from ExoGibbs develop commit
4a345444c26784559cac327a50ace088530f14f2, before the optional finite-log
independent audit. Existing chemical-source and equilibrium-impact receipts
pin the old audit SHA256. The historical resolver itself is also retained
because later receipts pin its original bytes. The VO2 revalidation also
pins its verifier, whose original source is retained here.

The current resolver checks each requested historical hash before returning
these files. Old receipts, numerical outputs, and acceptance claims are
unchanged. This source snapshot does not represent a new model execution.
