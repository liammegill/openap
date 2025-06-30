"""
Provides test for the addon bada3
"""

__author__ = "Liam Megill"
__email__ = "liam.megill@dlr.de"

# imports
import pytest
import numpy as np
from pyBADA.bada3 import Bada3Aircraft
from pyBADA.atmosphere import theta, sigma, delta
from openap.addon import bada3


# required fixtures
# pylint: disable=redefined-outer-name
@pytest.fixture
def mass() -> float:
    """Create mass fixture."""
    ac_model = bada3.load_bada3("BZJT", "DUMMY")
    return ac_model.MREF

@pytest.fixture
def tas() -> float:
    """Create TAS fixture."""
    return 150.0

@pytest.fixture
def alt() -> float:
    """Create altitude fixture."""
    return 10000.0

@pytest.fixture
def pybada_ac() -> Bada3Aircraft:
    """Create pyBADA aircraft fixture."""
    return Bada3Aircraft("DUMMY", "BZJT")

@pytest.fixture
def drag():
    """Initialises Drag class for a given aircraft."""
    return bada3.Drag("BZJT", "DUMMY")

@pytest.fixture
def thrust():
    """Initialises Thrust class for a given aircraft."""
    return bada3.Thrust("BZJT", "DUMMY")

@pytest.fixture
def ff():
    """Initialises FuelFlow class for a given aircraft."""
    return bada3.FuelFlow("BZJT", "DUMMY")


class TestDrag:
    """Tests class Drag."""

    def test_clean_drag(self, drag, pybada_ac, mass, tas, alt):
        """Compare OpenAP and pyBADA clean drag."""

        # OpenAP clean drag
        d_openap = drag.clean(mass, tas, alt)

        # pyBADA clean drag
        th = theta(alt * 0.3048, 0.0)
        d = delta(alt * 0.3048, 0.0)
        sig = sigma(th, d)
        cl = pybada_ac.CL(sig, mass, tas * 0.514)
        cd = pybada_ac.CD(cl, "CR")
        d_pybada = pybada_ac.D(sig, tas * 0.514, cd)

        # comparison
        assert np.isclose(d_openap, d_pybada, rtol=0.1)

    @pytest.mark.parametrize("phase", ["AP", "LD"])
    def test_nonclean_drag(self, drag, pybada_ac, mass, tas, alt, phase):
        """Compare OpenAP and pyBADA non-clean drag."""
        landing_gear = phase == "LD"

        # OpenAP non-clean drag
        d_openap = drag.nonclean(
            mass, tas, alt, landing_gear=landing_gear, phase=phase
        )

        # pyBADA non-clean drag
        th = theta(alt * 0.3048, 0.0)
        d = delta(alt * 0.3048, 0.0)
        sig = sigma(th, d)
        cl = pybada_ac.CL(sig, mass, tas * 0.514)
        cd = pybada_ac.CD(cl, phase)
        d_pybada = pybada_ac.D(sig, tas * 0.514, cd)

        # comparison
        assert np.isclose(d_openap, d_pybada, rtol=0.1)

    def test_incorrect_phase(self, drag, mass, tas, alt):
        """Test incorrect phase in nonclean drag."""
        with pytest.raises(ValueError):
            drag.nonclean(mass, tas, alt, phase="NA")


class TestThrust:
    """Tests class Thrust."""

    def test_climb_thrust(self, thrust, pybada_ac, tas, alt):
        """Compare OpenAP and pyBADA climb thrust (identical to takeoff)."""
        t_openap = thrust.climb(tas, alt)
        t_pybada = pybada_ac.Thrust(alt * 0.3048, 0.0, "MCMB", tas * 0.514, "CR")
        assert np.isclose(t_openap, t_pybada, rtol=0.1)

    def test_cruise_thrust(self, thrust, pybada_ac, tas, alt):
        """Compare OpenAP and pyBADA cruise thrust."""
        t_openap = thrust.cruise(tas, alt)
        t_pybada = pybada_ac.Thrust(alt * 0.3048, 0.0, "MCRZ", tas * 0.514, "CR")
        assert np.isclose(t_openap, t_pybada, rtol=0.1)

    @pytest.mark.parametrize("phase", ["CR", "AP", "LD"])
    def test_descent_thrust(self, thrust, pybada_ac, tas, alt, phase):
        """Compare OpenAP and pyBADA descent thrust."""
        t_openap = thrust.idle(tas, alt, config=phase)
        t_pybada = pybada_ac.Thrust(alt * 0.3048, 0.0, "LIDL", tas * 0.514, phase)
        assert np.isclose(t_openap, t_pybada, rtol=0.1)


class TestFuelFlow:
    """Tests class FuelFlow."""

    def test_nominal_ff(self, ff, pybada_ac, mass, tas, alt):
        """Compare OpenAP and pyBADA nominal fuel flow."""

        # OpenAP
        ff_openap = ff.nominal(mass, tas, alt)

        # pyBADA
        th = theta(alt * 0.3048, 0.0)
        d = delta(alt * 0.3048, 0.0)
        sig = sigma(th, d)
        cl = pybada_ac.CL(sig, mass, tas * 0.514)
        cd = pybada_ac.CD(cl, "CR")
        d_pybada = pybada_ac.D(sig, tas * 0.514, cd)
        t_pybada = d_pybada
        ff_pybada = pybada_ac.ffnom(tas * 0.514, t_pybada)
        # NOTE: pyBADA has no nominal fuel flow in their ff function
        # also, their climb thrust is not consistent with BADA3 documentation

        # comparison
        assert np.isclose(ff_openap, ff_pybada, rtol=1e-3)

    def test_enroute_ff(self, ff, pybada_ac, mass, tas, alt):
        """Compare OpenAP and pyBADA enroute fuel flow."""

        # OpenAP
        ff_openap = ff.enroute(mass, tas, alt)

        # pyBADA
        th = theta(alt * 0.3048, 0.0)
        d = delta(alt * 0.3048, 0.0)
        sig = sigma(th, d)
        cl = pybada_ac.CL(sig, mass, tas * 0.514)
        cd = pybada_ac.CD(cl, "CR")
        d_pybada = pybada_ac.D(sig, tas * 0.514, cd)
        t_pybada = d_pybada
        ff_pybada = pybada_ac.ff(
            alt * 0.3048, tas * 0.514, t_pybada, flightPhase="Cruise"
        )

        # comparison
        assert np.isclose(ff_openap, ff_pybada, rtol=1e-3)

    def test_idle_ff(self, ff, pybada_ac, mass, tas, alt):
        """Compare OpenAP and pyBADA idle fuel flow."""
        ff_openap = ff.idle(mass, tas, alt)
        ff_pybada = pybada_ac.ff(
            alt * 0.3048, tas * 0.514, None, config="CR", flightPhase="Descent"
        )
        assert np.isclose(ff_openap, ff_pybada, rtol=1e-3)

    def test_approach_ff(self, ff, pybada_ac, mass, tas, alt):
        """Compare OpenAP and pyBADA approach fuel flow."""

        # OpenAP
        ff_openap = ff.approach(mass, tas, alt)

        # pyBADA
        th = theta(alt * 0.3048, 0.0)
        d = delta(alt * 0.3048, 0.0)
        sig = sigma(th, d)
        cl = pybada_ac.CL(sig, mass, tas * 0.514)
        cd = pybada_ac.CD(cl, "CR")
        d_pybada = pybada_ac.D(sig, tas * 0.514, cd)
        tnom_pybada = d_pybada
        ff_pybada = pybada_ac.ff(
            alt * 0.3048, tas * 0.514, tnom_pybada,
            config="AP", flightPhase="Descent"
        )

        # comparison
        assert np.isclose(ff_openap, ff_pybada, rtol=1e-3)
