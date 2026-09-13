# orbital-mechanics

A Python library for orbital mechanics, mission design, and trajectory
optimization, built from first principles as a learning and portfolio
project. Every algorithm is implemented and unit-tested against known
reference solutions rather than wrapped from an existing library.

## Status

**Early / in progress.** Currently implemented:
- Cartesian state <-> classical orbital element conversion
- ECI <-> perifocal (PQW) frame transform

Next up: universal-variable Lambert solver, patched-conic interplanetary
transfer, launch-window (porkchop plot) analysis, validated against a real
past interplanetary mission.

## Installation (development)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -v
```

## Usage

```python
from orbital_mechanics import State

s = State(position=[-6045, -3490, 2500], velocity=[-3.457, 6.618, 2.533])
elements = s.to_orbit_elements()
print(elements)
```

## Why build this instead of using an existing library?

Libraries like poliastro/hapsira and Orekit already implement this scope
thoroughly. This project exists to build deep, first-hand understanding of
the algorithms and to practice production-quality software engineering
(testing, packaging, CI) -- not to replace those tools.
