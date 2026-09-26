# polaris

**P**atched-conic **O**rbital **L**ambert **A**nd **R**apid **I**nterplanetary **S**olver

A fast, first-principles Python astrodynamics library for **preliminary interplanetary trajectory design**, multi-gravity-assist (MGA) mission planning, and $\Delta v$ budgeting. 

Every algorithm is implemented from fundamental Keplerian mechanics, tested against published textbook references (Curtis, Vallado) and validated against historical interplanetary missions (Mars 2020 Perseverance, MESSENGER).

---

## Features

- **Core Orbital State Representations**:
  - `State`: Cartesian position/velocity vectors with input validation and immutability.
  - Robust conversion between Cartesian states and classical orbital elements ($a, e, i, \Omega, \omega, \nu$) using $\text{atan2}$ geometry and singularity guards.
  - ECI $\leftrightarrow$ Perifocal (PQW) reference frame transformations.
- **Universal-Variable Lambert Solver**:
  - Solves the two-point orbital boundary value problem for elliptic, parabolic, and hyperbolic transfers (Vallado Algorithm 58 / Curtis Section 5.3).
  - Hybrid Newton-Raphson iteration with bisection safeguard and negative-$y$ correction.
- **Ephemeris Retrieval**:
  - High-precision planetary state vectors via JPL Horizons API (`astroquery.jplhorizons`) with ISO date/JD parsing and LRU caching.
- **Patched-Conic Interplanetary Transfers**:
  - Single-leg interplanetary transfers calculating $C_3$, hyperbolic excess velocities ($\mathbf{v}_\infty$), and parking orbit departure/capture $\Delta v$ budgets.
  - 2D Porkchop plot grid analysis for launch window optimization.
- **Gravity Assists & Maneuvers**:
  - Hyperbolic gravity-assist kinematics: turn angle $\delta$, periapsis radius $r_p$ and altitude $h_p$, maximum safe deflection, and powered flyby $\Delta v_{\text{peri}}$.
  - Deep-space maneuver (DSM) two-subleg trajectory arc solver.
- **Multi-Leg Trajectory Assembly & Optimization**:
  - `solve_mga_trajectory()`: Assemble complex multi-gravity-assist sequences (e.g. Earth–Earth–Venus–Venus–Mercury–Mercury–Mercury–Mercury) with intermediate flybys and DSMs.
  - `optimize_mga_epochs()`: Numerical encounter epoch optimizer with cubic spline ephemeris pre-fetching.
- **NASA GMAT Export Bridge**:
  - Generates ready-to-run GMAT (.script) files to seed high-fidelity N-body numerical integration.

---

## Installation

```bash
git clone https://github.com/ishaansharma2903/polaris.git
cd polaris
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Classical Orbital Elements
```python
import numpy as np
from polaris import State, MU_EARTH

state = State(position=[-6045, -3490, 2500], velocity=[-3.457, 6.618, 2.533])
elements = state.to_orbit_elements(mu=MU_EARTH)

print(f"Semi-major axis: {elements.a:.1f} km")
print(f"Eccentricity: {elements.e:.4f}")
print(f"Inclination: {np.degrees(elements.i):.2f} deg")
```

### 2. Universal-Variable Lambert Solver
```python
import numpy as np
from polaris import solve_lambert, MU_EARTH

r1 = np.array([5000.0, 10000.0, 2100.0])
r2 = np.array([-14600.0, 2500.0, 7000.0])
tof = 3600.0  # seconds

v1, v2 = solve_lambert(r1, r2, tof, mu=MU_EARTH, prograde=True)
print("Departure velocity:", v1)
print("Arrival velocity:", v2)
```

### 3. Interplanetary Transfer & Porkchop Analysis
```python
from polaris import solve_single_leg, generate_porkchop

# Single leg transfer (Mars 2020 Perseverance launch window)
transfer = solve_single_leg(
    departure_body="earth",
    arrival_body="mars",
    departure_epoch="2020-07-30",
    arrival_epoch="2021-02-18",
)
print(f"Launch C3: {transfer.c3:.2f} km^2/s^2")
print(f"Arrival v_inf: {transfer.v_inf_arr_mag:.2f} km/s")

# Generate Porkchop grid
porkchop = generate_porkchop(
    departure_body="earth",
    arrival_body="mars",
    departure_window=("2020-07-01", "2020-08-31"),
    arrival_window=("2021-01-01", "2021-04-30"),
    step_dep_days=2.0,
    step_arr_days=2.0,
)
print("Minimum C3 in window:", np.nanmin(porkchop.c3))
```

### 4. Multi-Gravity-Assist (MGA) Sequence & GMAT Bridge
```python
from polaris import solve_mga_trajectory, export_to_gmat_script

bodies = ["earth", "venus", "mercury"]
epochs = ["2004-08-03", "2005-04-01", "2005-10-01"]

traj = solve_mga_trajectory(bodies=bodies, epochs=epochs)
print(f"Total mission duration: {traj.tof_total_days:.1f} days")

# Export to GMAT script for N-body propagation
gmat_script = export_to_gmat_script(traj, output_path="mga_mission.script", spacecraft_name="MessengerProbe")
```

---

## Running Tests

```bash
pytest -v --cov=polaris
```

All 72 tests pass with 99% test coverage across all modules.

---

## License

MIT License.
