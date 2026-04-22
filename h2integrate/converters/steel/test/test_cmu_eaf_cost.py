import pytest
import openmdao.api as om
from pytest import fixture

from h2integrate.converters.steel.cmu_eaf_cost import CMUElectricArcFurnaceCostModel


@fixture
def cost_config():
    config = {
        "model_inputs": {
            "cost_parameters": {
                "maintenance_cost_rate": 0.045,
                "mean_annual_wage": 66173,
                "mean_hourly_wage": 31.82,
                "eaf_labor_required_per_tLS": 4 / 20,
                "cost_year": 2022,
            }
        }
    }
    return config


@fixture
def plant_config():
    plant_config = {
        "plant": {
            "plant_life": 30,
            "simulation": {
                "n_timesteps": 8760,
                "dt": 3600,
            },
        },
        "finance_parameters": {
            "cost_adjustment_parameters": {
                "cost_year_adjustment_inflation": 0.025,
                "target_dollar_year": 2022,
            }
        },
    }
    return plant_config


@pytest.mark.unit
def test_cmu_eaf_cost(cost_config, plant_config, subtests):
    prob = om.Problem()

    cost = CMUElectricArcFurnaceCostModel(
        plant_config=plant_config, tech_config=cost_config, driver_config={}
    )

    prob.model.add_subsystem("cost", cost, promotes=["*"])
    prob.setup()

    ### TODO: determine how to handle oversizing vs production value
    prob.set_val("rated_steel_production", 2200000, units="t/year")

    prob.run_model()

    with subtests.test("capex"):
        assert pytest.approx(prob.get_val("CapEx", units="USD"), rel=1e-9) == 762839961.0691428

    with subtests.test("opex"):
        assert pytest.approx(prob.get_val("OpEx", units="USD/year"), rel=1e-9) == 48328598.24811143
