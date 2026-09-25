"""Shared Gaussian-cgs constants adopted by the manuscript reference calculation.

Electron mass is the CODATA2022 recommended value, converted from kg to g;
elementary charge uses the manuscript's conventional Coulomb-to-statcoulomb
conversion of the SI elementary charge. The charge denotes a positive
magnitude (electron charge is -E_ESU). Values are centralised so emission and
propagation cannot silently use different normalisations.

Sources: https://physics.nist.gov/cuu/pdf/wall_2022.pdf and
https://www.nist.gov/pml/special-publication-330/sp-330-section-2 .
Mass uncertainty and any convention/conversion uncertainty are not randomised
by this module; precision budgets requiring them should propagate them as
shared physical-constant inputs rather than independent per-module errors.
"""

E_ESU = 4.803204712570263e-10  # charge magnitude [statC]
M_E = 9.1093837139e-28  # electron mass [g]
C_CGS = 2.99792458e10  # speed of light [cm/s]
# SI speed of light [m/s]; only for the Faraday phase 2 (c/nu)^2 varphi with the
# wavelength in metres (syncmoments.model.phase), never for the emission kernels.
C_SI_M = 2.99792458e8

__all__ = ["E_ESU", "M_E", "C_CGS", "C_SI_M"]
