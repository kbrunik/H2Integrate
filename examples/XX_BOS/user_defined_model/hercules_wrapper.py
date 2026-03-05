from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


try:
    from hercules.hercules_model import HerculesModel
except ImportError:
    HerculesModel = None


@define(kw_only=True)
class HerculesConfig(BaseConfig):
    temp: int = field(default=0.0)
    hercules_config: dict = field(default={})


class HerculesWrapper(PerformanceModelBaseClass):
    def initialize(self):
        super().initialize()
        self.commodity = "electricity"
        self.commodity_amount_units = "kW*h"
        self.commodity_rate_units = "kW"

        if HerculesModel is None:
            raise ImportError(
                "The `hercules` package is required to use the performance model. "
                "Install it via:\n"
                "pip install git+https://github.com/NatLabRockies/hercules.git"
            )

    def setup(self):
        super().setup()

        self.config = HerculesConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )

        for herc_tech_cnfg in self.config.hercules_config.values():
            if (
                herc_tech_cnfg["component_type"] == "BatterySimple"
                or herc_tech_cnfg["component_type"] == "BatteryLithiumIon"
            ):
                self.add_input(
                    "max_charge_rate",
                    val=herc_tech_cnfg["charge_rate"],
                    units="kW",
                    desc="Battery charge rate",
                )

                self.add_input(
                    "storage_capacity",
                    val=herc_tech_cnfg["energy_capacity"],
                    units="kW*h",
                    desc="Battery storage capacity",
                )

            if herc_tech_cnfg["component_type"] == "OpenCycleGasTurbine":
                self.add_input(
                    "natural_gas_system_capacity",
                    val=herc_tech_cnfg["rated_capacity"],
                    units="kW",
                    desc="Natural gas plant rated capacity in kW",
                )

            if herc_tech_cnfg["component_type"] == "SolarPySAMPVWatts":
                self.add_input(
                    "pv_capacity_DC",
                    val=herc_tech_cnfg["system_capacity"],
                    units="kW",
                    desc="PV rated capacity in DC",
                )
                self.add_output(
                    "pv_capacity_AC", val=0.0, units="kW", desc="PV rated capacity in AC"
                )

            if (
                herc_tech_cnfg["component_type"] == "WindFarm"
                or herc_tech_cnfg["component_type"] == "WindFarmSCADAPower"
            ):
                # not yet supported as a design variable
                pass

            if herc_tech_cnfg["component_type"] == "ElectrolyzerPlant":
                # not yet supported for a design variable
                pass

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # NOTE: need to add a controller, which is a class in hercules

        pass
