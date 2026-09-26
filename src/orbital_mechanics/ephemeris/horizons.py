"""
ephemeris/horizons.py

Interface to JPL Horizons for planetary ephemeris retrieval.
Queries positions and velocities at arbitrary epochs, converts to standard
library units (km, km/s), and returns immutable State instances.
"""
from functools import lru_cache
from typing import Dict, Union, List
import datetime
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from orbital_mechanics.constants import AU, DAY_TO_SEC
from orbital_mechanics.core.state import State

# Standard body aliases -> JPL Horizons target IDs
# Planet centers (e.g. 399 for Earth) or barycenters
BODY_MAP: Dict[str, str] = {
    "sun": "10",
    "sol": "10",
    "mercury": "199",
    "venus": "299",
    "earth": "399",
    "moon": "301",
    "mars": "499",
    "jupiter": "599",
    "saturn": "699",
    "uranus": "799",
    "neptune": "899",
    "pluto": "999",
    # Barycenters
    "mercury_barycenter": "1",
    "venus_barycenter": "2",
    "earth_barycenter": "3",
    "mars_barycenter": "4",
    "jupiter_barycenter": "5",
    "saturn_barycenter": "6",
    "uranus_barycenter": "7",
    "neptune_barycenter": "8",
    "pluto_barycenter": "9",
}

CENTER_MAP: Dict[str, str] = {
    "sun": "@10",
    "sol": "@10",
    "ssb": "@0",
    "barycenter": "@0",
    "solar_system_barycenter": "@0",
    "earth": "@399",
    "emb": "@3",
    "mars": "@499",
    "venus": "@299",
    "mercury": "@199",
}


def parse_epoch(epoch: Union[float, int, str, datetime.date, datetime.datetime, Time]) -> float:
    """
    Parse an epoch input into a Julian Date (JD) float.

    Supported input types:
    - float / int (assumed to be a Julian Date, e.g. 2451545.0)
    - ISO date string (e.g. '2004-08-03', '2004-08-03 06:15:00')
    - datetime.date / datetime.datetime
    - astropy.time.Time

    Returns:
        float: Julian Date in days.
    """
    if isinstance(epoch, Time):
        return float(epoch.jd)
    if isinstance(epoch, (int, float)):
        return float(epoch)
    if isinstance(epoch, datetime.datetime):
        return float(Time(epoch).jd)
    if isinstance(epoch, datetime.date):
        dt = datetime.datetime.combine(epoch, datetime.time())
        return float(Time(dt).jd)
    if isinstance(epoch, str):
        try:
            return float(epoch)
        except ValueError:
            return float(Time(epoch).jd)
    raise TypeError(f"Unsupported epoch type: {type(epoch)}")


def _resolve_body_id(body: Union[str, int]) -> str:
    """Resolve a body name or ID to a Horizons target identifier string."""
    if isinstance(body, int):
        return str(body)
    body_str = str(body).strip().lower()
    return BODY_MAP.get(body_str, str(body))


def _resolve_center_id(center: Union[str, int]) -> str:
    """Resolve an observer center location to a Horizons location string."""
    if isinstance(center, int):
        return f"@{center}"
    center_str = str(center).strip().lower()
    if center_str.startswith("@"):
        return center_str
    if center_str in CENTER_MAP:
        return CENTER_MAP[center_str]
    return f"@{center_str}"


@lru_cache(maxsize=4096)
def _query_horizons_cached(
    target_id: str,
    epoch_jd: float,
    location: str,
    refplane: str,
) -> tuple:
    """
    Query JPL Horizons and return raw vector tuple (x, y, z, vx, vy, vz).
    Cached via LRU to prevent duplicate remote queries during grid searches.
    """
    obj = Horizons(id=target_id, location=location, epochs=epoch_jd)
    vectors = obj.vectors(refplane=refplane)
    if len(vectors) == 0:
        raise ValueError(f"No ephemeris data returned for target={target_id} at epoch={epoch_jd}")
    
    row = vectors[0]
    return (
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
        float(row["vx"]),
        float(row["vy"]),
        float(row["vz"]),
    )


def get_body_state(
    body: Union[str, int],
    epoch: Union[float, int, str, datetime.date, datetime.datetime, Time],
    center: Union[str, int] = "sun",
    refplane: str = "ecliptic",
) -> State:
    """
    Retrieve the Cartesian State (position & velocity) of a celestial body
    at a given epoch from JPL Horizons.

    Units:
        position: km
        velocity: km/s

    Parameters:
        body: Body name (e.g. 'earth', 'venus', 'mercury', 'mars') or NAIF/Horizons ID (e.g. 399).
        epoch: Epoch as Julian Date (float), ISO date string, datetime object, or astropy Time.
        center: Observer coordinate origin (default: 'sun' / '@10').
        refplane: Reference plane, either 'ecliptic' (default) or 'frame' (ICRF/equatorial).

    Returns:
        State: State instance containing position vector (km) and velocity vector (km/s).
    """
    target_id = _resolve_body_id(body)
    location = _resolve_center_id(center)
    epoch_jd = parse_epoch(epoch)

    x_au, y_au, z_au, vx_aud, vy_aud, vz_aud = _query_horizons_cached(
        target_id=target_id,
        epoch_jd=epoch_jd,
        location=location,
        refplane=refplane,
    )

    pos_km = np.array([x_au, y_au, z_au], dtype=float) * AU
    vel_kms = np.array([vx_aud, vy_aud, vz_aud], dtype=float) * (AU / DAY_TO_SEC)

    return State(position=pos_km, velocity=vel_kms)


def get_body_states_batch(
    body: Union[str, int],
    epochs: Union[List[Union[float, int, str, datetime.date, datetime.datetime, Time]], np.ndarray],
    center: Union[str, int] = "sun",
    refplane: str = "ecliptic",
) -> List[State]:
    """
    Retrieve Cartesian States of a celestial body across multiple epochs in a
    single batch Horizons query.

    Parameters:
        body: Body name or ID.
        epochs: Sequence of epochs.
        center: Observer coordinate origin (default: 'sun').
        refplane: Reference plane (default: 'ecliptic').

    Returns:
        List[State]: List of State instances corresponding to each epoch.
    """
    parsed_jds = [parse_epoch(ep) for ep in epochs]
    if len(parsed_jds) == 0:
        return []
    if len(parsed_jds) == 1:
        return [get_body_state(body, parsed_jds[0], center=center, refplane=refplane)]

    target_id = _resolve_body_id(body)
    location = _resolve_center_id(center)

    obj = Horizons(id=target_id, location=location, epochs=parsed_jds)
    vectors = obj.vectors(refplane=refplane)
    if len(vectors) != len(parsed_jds):
        raise ValueError(f"Horizons returned {len(vectors)} states for {len(parsed_jds)} requested epochs.")

    states: List[State] = []
    for row in vectors:
        pos_km = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float) * AU
        vel_kms = np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])], dtype=float) * (AU / DAY_TO_SEC)
        states.append(State(position=pos_km, velocity=vel_kms))
    return states


def clear_ephemeris_cache() -> None:
    """Clear in-memory cached Horizons query results."""
    _query_horizons_cached.cache_clear()
