from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gt_zero, must_equal
from h2integrate.core.model_baseclasses import (
    CostModelBaseClass,
    CostModelBaseConfig,
    PerformanceModelBaseClass,
)


@define(kw_only=True)
class TigerTidalPerformanceConfig(BaseConfig):
    """
    Configuration class for TigerTidalPerformanceModel.

    Inputs you want the users to be able to set from the `tech_config.yaml`

    Args:
        device_rating_kw (float): Rated power of the MHK device [kW]
        num_devices (int): Number of MHK tidal devices in the system
        rotor_radius (float): Rotor radius of the tidal energy device [m]

        Be sure to update this with all of the new parameters you want to be
        able to set from the `tech_config.yaml`
        file when instantiating the model, and to add validators as needed.
        You can also add optional parameters with default values.

    """

    # ADD PERFORMANCE INPUTS HERE
    device_rating_kw: float = field(validator=gt_zero)
    num_devices: int = field(validator=gt_zero)
    rotor_radius: float = field(validator=gt_zero)
    water_ph: float = field(default=8.0, validator=gt_zero)


### Tiger tidal performance model
class TigerTidalPerformanceModel(PerformanceModelBaseClass):
    """An OpenMDAO component for the Tiger tidal performance model.
    It takes tidal parameters as input and outputs power generation data.
    """

    # This is considered an hourly timestep
    _time_step_bounds = (
        3600,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def initialize(self):
        super().initialize()
        self.commodity = "electricity"
        self.commodity_rate_units = "kW"
        self.commodity_amount_units = "kW*h"

    def setup(self):
        super().setup()
        self.config = TigerTidalPerformanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )

        #### Tidal Resource ####
        self.add_input(
            "tidal_velocity",
            val=0.0,
            shape=self.n_timesteps,
            units="m/s",
        )

        #### Tidal Device Parameters ####
        # rotor radius, single turbine capacity, number of turbines,
        self.add_input(
            "rotor_radius",
            val=self.config.rotor_radius,
            units="m",
            desc="Rotor radius of the tidal energy device",
        )

        self.add_input(
            "num_devices",
            val=self.config.num_devices,
            units="unitless",
            desc="Number of tidal devices in the system",
        )

        self.add_input(
            "device_rating",
            val=self.config.device_rating_kw,
            units="kW",
            desc="Rated power of the tidal energy device",
        )

        #### Can add other outputs if you want them
        # self.add_output(
        #     "power_curve",
        #     units="kW",
        #     desc="Power curve of the tidal energy device as a function of tidal velocity",
        # )

    def compute(self, inputs, outputs):
        # assign resource to tidal model
        inputs["tidal_velocity"]

        # calculate system capacity
        system_capacity_kw = inputs["num_devices"][0] * inputs["device_rating"][0]

        # run the necessary calculations

        ##### Add tiger performance model calculations here #####

        ### outputs from the model
        outputs["electricity_out"] = 0  # Add timeseries of the power output here
        outputs["rated_electricity_production"] = system_capacity_kw

        outputs["total_electricity_produced"] = outputs["electricity_out"].sum() * (self.dt / 3600)
        outputs["annual_electricity_produced"] = 0

        outputs["capacity_factor"] = (
            self.system_model.Outputs.capacity_factor / 100
        )  # divide by 100 to make it unitless


@define(kw_only=True)
class TigerTidalCostConfig(CostModelBaseConfig):
    """Add summary of config here."""

    ##### ADD COST MODEL INPUTS HERE

    # if the cost year has to be a specific year update it here
    cost_year: int = field(default=2024, converter=int, validator=must_equal(2024))


class TigerTidalCostModel(CostModelBaseClass):
    """An OpenMDAO component for the Tiger tidal cost model.
    It takes tidal device parameters as input and outputs cost data.
    """

    def setup(self):
        super().setup()
        self.config = TigerTidalCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            additional_cls_name=self.__class__.__name__,
        )

        ##### Add cost model inputs here #####

    def compute(self, inputs, outputs):
        ##### Add tiger cost model calculations here #####

        ### outputs from the model
        outputs["CapEx"] = 0
        outputs["OpEx"] = 0
