"""
tests/transfers/test_porkchop.py

Unit and validation tests for launch window porkchop plots.
"""
import pytest
import numpy as np

from polaris.constants import (
    MU_EARTH,
    MU_MARS,
    R_EARTH,
    R_MARS,
)
from polaris.transfers.porkchop import PorkchopResult, generate_porkchop


def test_generate_porkchop_earth_mars_2020():
    """
    Validate porkchop plot generation across the Mars 2020 Perseverance launch window.
    """
    porkchop = generate_porkchop(
        departure_body="earth",
        arrival_body="mars",
        dep_epoch_range=("2020-07-15", "2020-08-15"),
        arr_epoch_range=("2021-02-01", "2021-03-01"),
        num_dep=8,
        num_arr=8,
        r_park_dep=R_EARTH + 200.0,
        mu_dep=MU_EARTH,
        r_park_arr=R_MARS + 250.0,
        mu_arr=MU_MARS,
    )

    assert isinstance(porkchop, PorkchopResult)
    assert porkchop.c3.shape == (8, 8)
    assert porkchop.tof_days.shape == (8, 8)
    assert porkchop.total_delta_v.shape == (8, 8)

    min_c3_val, best_dep, best_arr = porkchop.min_c3()
    assert 12.0 < min_c3_val < 18.0

    min_dv_val, best_dep_dv, best_arr_dv = porkchop.min_delta_v()
    assert 5.0 < min_dv_val < 8.0


def test_generate_porkchop_input_validation():
    """Verify input validation for invalid epoch ranges and grid sizes."""
    with pytest.raises(ValueError, match="dep_epoch_range end"):
        generate_porkchop("earth", "mars", ("2020-08-15", "2020-07-15"), ("2021-02-01", "2021-03-01"))

    with pytest.raises(ValueError, match="arr_epoch_range end"):
        generate_porkchop("earth", "mars", ("2020-07-15", "2020-08-15"), ("2021-03-01", "2021-02-01"))

    with pytest.raises(ValueError, match="at least 2"):
        generate_porkchop("earth", "mars", ("2020-07-15", "2020-08-15"), ("2021-02-01", "2021-03-01"), num_dep=1)
