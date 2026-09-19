"""
constants.py

Physical constants used throughout the library. Centralized here so a
value is never redefined (and potentially made inconsistent) across
multiple modules.

Units: km, km/s, km^3/s^2 unless otherwise noted, consistent throughout
the library.
"""

# --- Numerical tolerances -------------------------------------------------
EPSILON = 1e-8  # tolerance for dimensionless near-zero comparisons

# --- Gravitational parameters (mu = G*M), km^3/s^2 ------------------------
# Source: JPL DE430 / standard astrodynamics references (Curtis, Vallado)
MU_SUN = 1.32712440018e11
MU_MERCURY = 22032.0
MU_VENUS = 324858.592
MU_EARTH = 398600.4418
MU_MARS = 42828.37

# --- Body radii (equatorial), km -------------------------------------------
R_SUN = 695700.0
R_MERCURY = 2439.7
R_VENUS = 6051.84
R_EARTH = 6378.137
R_MARS = 3396.2

# --- Distance & Time conversions ------------------------------------------
AU = 1.495978707e8  # km, one astronomical unit
DAY_TO_SEC = 86400.0  # seconds in one Julian day (86400 s)

# --- Earth oblateness -----------------------------------------------------
J2_EARTH = 1.08263e-3  # dimensionless, used once perturbations are added
