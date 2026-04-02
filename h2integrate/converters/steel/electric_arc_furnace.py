"""Electric Arc Furnace performance model based on CMU decarbSTEEL EAF Model"""

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass
from h2integrate.converters.steel.steel_eaf_base import ElectricArcFurnacePerformanceBaseConfig

from attrs import field, define

from openmdao.utils import units
# data[data_col] = units.convert_units(data[data_col], orig_units, desired_units)
 
# NOTE: where should these live?
# TODO: core > constants.py 
# Conversion factors , Rows 133-143
MMbtu_to_kWh = 293.0  # kWh, 'Model Inputs & Outputs!C134'
MMbtu_to_MJ = 1055.0  # MJ, 'Model Inputs & Outputs!C135'
kWh_to_kJ = 3600.0  # kJ, 'Model Inputs & Outputs!C136'
kWh_to_J = 3600000.0  # J, 'Model Inputs & Outputs!C137'
MJ_H2_to_kg_H2 = 0.01  # kg H2, H2 lower heating value = 120.0 kg/MJ, 'Model Inputs & Outputs!C138'
kmol_H2_to_kg_H2 = 2.02  # kg H2, 'Model Inputs & Outputs!C134'
kmol_NG_to_kg_NG = 16.04  # kg NG, approximated as CH4, 'Model Inputs & Outputs!C139'
kmol_NG_to_Nm3_NG = 1.39  # Nm^3 NG, approximated as CH4, 'Model Inputs & Outputs!C140'
kmol_NG_to_MMbtu_NG = 0.04  # MMBtu NG, approximated as CH4, 'Model Inputs & Outputs!C141'
kmol_NG_to_MMbtu = 0.79  # MMBtu, approximated as CH4, 'Model Inputs & Outputs!C142'

# CMU EAF model pythonization
# NOTE: values are largely in metric system, all tons = metric tons

@define
class CMUElectricArcFurnaceScrapOnlyPerformanceBaseConfig(BaseConfig):
    """Configuration baseclass for CMUElectricArcFurnaceScrapOnlyPerformanceComponent.

    Attributes:
        steel_production_rate_tonnes_per_hr (float): capacity of the steel processing plant
            in units of metric tonnes of steel produced per hour.
        water_density (float): water density in kg/m3 to use to calculate water volume
            from mass. Defaults to 1000.0
    """

    # steel_capacity_rate_tonnes_per_year: float = field(default=2.2)   # metric tons/year
    steel_production_rate_tonnes_per_year: float = field(default=2.0)   # metric tons/year
    steel_percent_carbon: float = field(default=0.1 / 100) # mass fraction C in steel out, 'Model Inputs & Outputs!B26'
    scrap_composition: dict = field(default={
            "Fe": 94.0 / 100,  # mass fraction Fe, 'Model Inputs & Outputs!B27'
            "SiO2": 1.0 / 100,  # mass fraction SiO2, 'Model Inputs & Outputs!B28'
        })
    # water_density: float = field(default=1000)  # kg/m3

class CMUElectricArcFurnaceScrapOnlyPerformanceComponent(PerformanceModelBaseClass):
    def initialize(self):
        super().initialize()
        self.commodity = "steel"
        self.commodity_rate_units = "t/h"
        self.commodity_amount_units = "t"

    def setup(self):
        super().setup()

        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]

        self.config = CMUElectricArcFurnaceScrapOnlyPerformanceBaseConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=True,
            additional_cls_name=self.__class__.__name__,
        )

        # TODO: remove
        # steel_capacity = 2.2 mTons/year, 'Model Inputs & Outputs!B11'
        # self.add_input(
        #     "system_capacity",
        #     val=self.config.steel_capacity_rate_tonnes_per_year,
        #     units='t/year',
        #     desc="Rated steel production capacity",
        # )

        # annual_production = 2.0 mTons/year, 'Model Inputs & Outputs!B12'
        self.add_input(
            "annual_production",
            val=self.config.steel_production_rate_tonnes_per_year,  # update
            units='t/year',
            desc="Actual steel production",
        )

        # NOTE: was going to add plant life = 40 years here but that seems to be part of the cost component in other models?

        # Add feedstock inputs and outputs, default to 0 --> set using feedstock component
        # (everything under "Input feedstocks for EAF Fed with Scrap Only")
        # oxygen, electricity, natural gas, electrodes, scrap steel, coal, burnt doloma, burnt lime

        feedstocks_to_units = {
            "oxygen":"m**3/t",
            "electricity":"kWh/t",
            "natural_gas":"MMBtu/t",
            "electrodes":"kg/t",
            "scrap":"t/t",
            "coal":"t/t",
            "doloma":"t/t",
            "lime":"t/t"
        }

        for feedstock, feedstock_units in feedstocks_to_units.items():
            self.add_input(
                f"{feedstock}_in",
                val=0.0,
                shape=n_timesteps,
                units=feedstock_units,
                desc=f"{feedstock} available for steel production",
            )
            self.add_output(
                f"{feedstock}_consumed",
                val=0.0,
                shape=n_timesteps,
                units=feedstock_units,
                desc=f"{feedstock} consumed for steel production",
            )

        #NOTE: depending on if outputs on per tLS or per tscrap basis, desc may need to be dynamically set?
        # Add outputs
        self.add_output(
            "mass_slag",  # == mass_slag_per_tscrap
            val=0.0,
            shape=n_timesteps, 
            units="kg/t",
            desc="Total unit of slag per unit of scrap",
        )

        self.add_output(
            "mass_MgO_slag",  # == mass_MgO_slag_per_tscrap
            val=0.0,
            shape=n_timesteps, 
            units="kg/t",
            desc="Total unit of MgO in slag per unit of scrap",
        )

        self.add_output(
            "mass_FeO_slag",  # == mass_FeO_slag_per_tscrap
            val=0.0,
            shape=n_timesteps, 
            units="kg/t",
            desc="Total unit of FeO in slag per unit of scrap",
        )

        self.add_output(
            "mass_Fe_to_FeO",  # == mass_Fe_to_FeO_tscrap
            val=0.0,
            shape=n_timesteps, 
            units="kg/t",
            desc="Total unit of Fe consumed to produce FeO per unit of scrap",
        )

        self.add_output(
            "mass_Fe_from_scrap",  # == mass_Fe_DRI_per_tscrap
            val=0.0,
            shape=n_timesteps, 
            units="kg/t",
            desc="Total unit of Fe from scrap per unit of scrap",
        )

        self.add_output(
            "mass_steel",  # == mass_steel_per_tscrap
            val=0.0,
            shape=n_timesteps, 
            units="kg/t",
            desc="Total unit of steel formed from EAF fed with scrap only per unit of scrap",
        )

    def compute(self, inputs, outputs):
        # Steel and scrap composition
        # NOTE update below 2 feedstock sections to get feedstocks appropriately based on updates in setup
        # get feedstocks

        # get feedstock usage rates
        # Including DRI in feed (assumed constants in feedstocks)
        natural_gas = 0.44  # MMBtu/ton steel, '5. Electric Arc Furnace!C32'
        electrodes = 2.00  # kg/ton steel, '5. Electric Arc Furnace!C33'

        eaf_scrap_only_output_dict = self.CMUElectricArcFurnaceScrapOnlyMassEnergyBalance()

        # Insert logic for if determining inputs based on desired steel production rate
        # or if determining steel production based on inputs
        if True:
            
        # set outputs
        # Input feedstocks for EAF Fed with Scrap Only
        outputs["oxygen"] = oxygen  # Total Nm^3 O2 per ton steel
        outputs["electricity"] = electricity  # Total kWh electricity per ton steel
        outputs["natural_gas"] = natural_gas  # Total MMBtu NG per ton steel
        outputs["electrodes"] = electrodes  # Total kg Electrodes per ton steel
        outputs["scrap steel"] = scrap_steel  # Total ton scrap per ton steel
        outputs["coal"] = coal  # Total ton coal per ton steel
        outputs["burnt doloma"] = burnt_doloma  # Total ton burnt doloma per ton steel
        outputs["burnt_lime"] = burnt_lime  # Total ton burnt lime per ton steel

        # Output for EAF Fed with Scrap only
        outputs["mass slag"] = mass_slag_per_tscrap  # Total kg slag per tscrap
        outputs["mass MgO slag"] = mass_MgO_slag_per_tscrap  # Total kg MgO in slag per tscrap
        outputs["mass FeO slag"] = mass_FeO_slag_per_tscrap  # Total kg FeO in slag per tscrap
        outputs["mass Fe to FeO"] = (
            mass_Fe_to_FeO_tscrap  # Total kg Fe consumed to produce FeO per tscrap
        )
        outputs["mass Fe from scrap"] = mass_Fe_DRI_per_tscrap  # Total kg Fe from scrap per tscrap
        outputs["mass steel"] = (
            mass_steel_per_tscrap  # Total kg Steel formed from scrap per ton scrap
        )

    def CMUElectricArcFurnaceScrapOnlyMassEnergyBalance(self):
        output_dict = {}
        # Including DRI in feed (assumed constants in feedstocks)
        natural_gas = 0.44  # MMBtu/ton steel, '5. Electric Arc Furnace!C32'
        electrodes = 2.00  # kg/ton steel, '5. Electric Arc Furnace!C33'

        # 12. EAF Mass & Energy Balance 
        # Essential Mass Summary
        # NOTE: Hardcoding mass_steel_stream and mass_basis_scrap for per ton liquid steel and per ton scrap basis calculations
        mass_steel_stream = 1000  # kg liquid steel, '12. EAF Mass & Energy Balance!D4'
        pct_carbon_steel = self.config.steel_percent_carbon  # % mass C, 'Model Inputs & Outputs!B26' > '12. EAF Mass & Energy Balance!D5'
        mass_iron_per_tLS = mass_steel_stream * (1 - pct_carbon_steel)  # kg Fe/ton liquid steel, '12. EAF Mass & Energy Balance!D7'
        mass_carbon_per_tLS = mass_steel_stream * pct_carbon_steel  # kg C/ton liquid steel, '12. EAF Mass & Energy Balance!D8'
        scrap_composition = self.config.scrap_composition   # % mass Fe and % mass SiO2, 'Model Inputs & Outputs!B27' & 'Model Inputs & Outputs!B28' 

        # Electric Arc Furnace Fed with Scrap Only - Mass Balance (Fe, C, O, MgO, SiO2, Al2O3, CaO)
        # NOTE: calculated per ton scrap (tsrap)
        mass_basis_scrap = 1000  # kg, '12. EAF Mass & Energy Balance!D47'
        mass_pct_SiO2_scrap = scrap_composition["SiO2"]  # % mass SiO2, 'Model Inputs & Outputs!B28' > '12. EAF Mass & Energy Balance!D48'
        mass_SiO2_scrap_per_tscrap = mass_basis_scrap * mass_pct_SiO2_scrap  # kg SiO2 per ton scrap, '12. EAF Mass & Energy Balance!D49'

        slag_B3 = 1.50  # basicity, kg CaO / (kg SiO2 + kg Al2O3), '12. EAF Mass & Energy Balance!D51'
        mass_SiO2_slag_per_tscrap = mass_SiO2_scrap_per_tscrap  # kg total SiO2 in slag per ton scrap, '12. EAF Mass & Energy Balance!D52'
        mass_Al2O3_slag_per_tscrap = 0.0 # kg Al2O3 in slag per ton scrap, '12. EAF Mass & Energy Balance!D53'
        mass_CaO_slag_per_tscrap = slag_B3 * (mass_SiO2_slag_per_tscrap + mass_Al2O3_slag_per_tscrap) # kg CaO in slag per ton scrap,  '12. EAF Mass & Energy Balance!D54'

        pct_MgO_slag = (12.0 / 100) # mass fraction MgO in slag, assumed input, '12. EAF Mass & Energy Balance!D56'
        pct_FeO_slag = (30.0 / 100) # mass fraction FeO in slag, assumed input, '12. EAF Mass & Energy Balance!D57'
        output_dict["mass_slag_per_tscrap"] = (
            mass_SiO2_slag_per_tscrap + mass_Al2O3_slag_per_tscrap + mass_CaO_slag_per_tscrap
        ) / (1 - pct_MgO_slag - pct_FeO_slag)  # kg slag per ton scrap, '12. EAF Mass & Energy Balance!D58'
        output_dict["mass_MgO_slag_per_tscrap"] = (pct_MgO_slag * output_dict["mass_slag_per_tscrap"])  # kg MgO in slag per ton scrap, '12. EAF Mass & Energy Balance!D59'
        output_dict["mass_FeO_slag_per_tscrap"] = (
            pct_FeO_slag * output_dict["mass_slag_per_tscrap"]
        )  # kg FeO in slag per ton scrap, '12. EAF Mass & Energy Balance!D60'
        moles_FeO_slag_per_tscrap = (
            output_dict["mass_FeO_slag_per_tscrap"] / 71.8
        )  # kmol FeO in slag per ton scrap, '12. EAF Mass & Energy Balance!D61'
        moles_Fe_to_FeO_tscrap = moles_FeO_slag_per_tscrap  # kmol Fe consumed to produce FeO per ton scrap, '12. EAF Mass & Energy Balance!D62'
        output_dict["mass_Fe_to_FeO_tscrap"] = (
            moles_Fe_to_FeO_tscrap * 55.80
        )  # kg Fe consumed to produce FeO per ton scrap, '12. EAF Mass & Energy Balance!D63'

        output_dict["mass_Fe_scrap_per_tscrap"] = (
            (mass_basis_scrap * scrap_composition["Fe"]) - output_dict["mass_Fe_to_FeO_tscrap"]
        )  # kg Fe mass from scrap per ton scrap, '12. EAF Mass & Energy Balance!D65'
        output_dict["mass_steel_per_tscrap"] = output_dict["mass_Fe_scrap_per_tscrap"] / (
            1 - pct_carbon_steel
        )  # kg steel formed from DRI + scrap per ton srap, '12. EAF Mass & Energy Balance!D66'

        # NOTE calculated per ton liquid steel (tLS)
        output_dict["mass_scrap_per_tLS"] = (
            mass_basis_scrap / output_dict["mass_steel_per_tscrap"]
        ) * 1000  # kg scrap per ton LS, '12. EAF Mass & Energy Balance!D69'
        mass_pct_SiO2_scrap = scrap_composition[
            "SiO2"
        ]  # mass fraction SiO2, '12. EAF Mass & Energy Balance!D70' > 'Model Inputs & Outputs!B28'
        mass_SiO2_scrap_per_tLS = (
            output_dict["mass_scrap_per_tLS"] * mass_pct_SiO2_scrap
        )  # kg SiO2 from scrap per ton LS, '12. EAF Mass & Energy Balance!D71'

        slag_B3 = (
            1.5  # basicity, kg CaO / (kg SiO2 + kg Al2O3), '12. EAF Mass & Energy Balance!D73'
        )
        mass_SiO2_slag_per_tLS = mass_SiO2_scrap_per_tLS  # total kg SiO2 in slag per ton LS, '12. EAF Mass & Energy Balance!D74'
        mass_Al2O3_slag_per_tLS = (
            0.0  # total kg Al2O3 in slag per ton LS, '12. EAF Mass & Energy Balance!D75'
        )
        mass_CaO_slag_per_tLS = slag_B3 * (
            mass_SiO2_slag_per_tLS + mass_Al2O3_slag_per_tLS
        )  # total kg CaO in slag per ton LS, '12. EAF Mass & Energy Balance!D76'

        pct_MgO_slag = (
            12.0 / 100
        )  # mass fraction MgO in slag, assumed input, '12. EAF Mass & Energy Balance!D78'
        pct_FeO_slag = (
            30.0 / 100
        )  # mass fraction FeO in slag, assumed input, '12. EAF Mass & Energy Balance!D79'
        output_dict["mass_slag_per_tLS"] = (
            mass_SiO2_slag_per_tLS + mass_Al2O3_slag_per_tLS + mass_CaO_slag_per_tLS
        ) / (
            1 - pct_MgO_slag - pct_FeO_slag
        )  # kg slag per ton LS, '12. EAF Mass & Energy Balance!D80'
        output_dict["mass_MgO_slag_per_tLS"] = (
            pct_MgO_slag * output_dict["mass_slag_per_tLS"]
        )  # total mass MgO in slag per ton LS, '12. EAF Mass & Energy Balance!D81'
        output_dict["mass_FeO_slag_per_tLS"] = (
            pct_FeO_slag * output_dict["mass_slag_per_tLS"]
        )  # total mass FeO in slag per ton LS, '12. EAF Mass & Energy Balance!D82'
        moles_FeO_slag_per_tLS = (
            output_dict["mass_FeO_slag_per_tLS"] / 71.80
        )  # moles FeO in slag per ton LS, '12. EAF Mass & Energy Balance!D83'
        moles_Fe_to_FeO_tLS = moles_FeO_slag_per_tLS  # moles Fe consumed to produce FeO in slag per ton LS, '12. EAF Mass & Energy Balance!D84'
        output_dict["mass_Fe_to_FeO_tLS"] = moles_Fe_to_FeO_tLS * 55.80  # kg Fe consumed to produce FeO in slag per ton LS, '12. EAF Mass & Energy Balance!D85'
        moles_O2_to_FeO_tLS = (
            moles_Fe_to_FeO_tLS * 0.5
        )  # moles O2 consumed to produce FeO in slag per ton LS, '12. EAF Mass & Energy Balance!D86'

        output_dict["mass_Fe_scrap_per_tLS"] = output_dict["mass_Fe_scrap_per_tscrap"] * (output_dict["mass_scrap_per_tLS"] / 1000)

        mass_C_ng_per_tLS = ((natural_gas * MMbtu_to_MJ) / 50.0) * (
            12.0 / 16.0
        )  # kg Carbon in NG per ton LS, '12. EAF Mass & Energy Balance!D88'
        pct_carbon_steel_tap = (
            3 / 100
        )  # mass fraction carbon input to EAF as % of steel tap mass, '12. EAF Mass & Energy Balance!D89'
        total_C_kg_per_tLS = (
            pct_carbon_steel_tap * 1000
        )  # kg total carbon input per tLS, '12. EAF Mass & Energy Balance!D90'
        mass_injected_carbon_per_tLS = total_C_kg_per_tLS  # additional carbon required / injected per tLS, '12. EAF Mass & Energy Balance!D91'
        moles_C_ng_per_tLS = (
            mass_C_ng_per_tLS / 12.0
        )  # kmol Carbon in NG blown out per ton LS, '12. EAF Mass & Energy Balance!D92'
        moles_O2_ng_per_tLS = (
            moles_C_ng_per_tLS * 1.0
        )  # kmol O2 needed to blow out NG per tLS, '12. EAF Mass & Energy Balance!D93'
        moles_CO2_ng_per_tLS = (
            moles_C_ng_per_tLS * 1.0
        )  # kmol CO2 formed per tLS, '12. EAF Mass & Energy Balance!D94'
        (moles_CO2_ng_per_tLS * 44.0)  # kg CO2 formed per tLS, '12. EAF Mass & Energy Balance!D95'
        moles_C_injected_per_tLS = (
            mass_injected_carbon_per_tLS - mass_carbon_per_tLS
        ) / 12.0  # kmol C per tLS, '12. EAF Mass & Energy Balance!D96'
        moles_O2_injected_per_tLS = (
            moles_C_injected_per_tLS * 0.5
        )  # kmol O2 needed to blow out C in injected Carbon, '12. EAF Mass & Energy Balance!D97'
        moles_CO_injected_per_tLS = (
            moles_C_injected_per_tLS  # kmol CO formed per tLS, '12. EAF Mass & Energy Balance!D98'
        )
        (
            moles_CO_injected_per_tLS * 28.0
        )  # kg CO formed per tLS, '12. EAF Mass & Energy Balance!D99'
        moles_O2_per_tLS = (
            moles_O2_ng_per_tLS + moles_O2_injected_per_tLS + moles_O2_to_FeO_tLS
        )  # kmol O2 required per tLS, '12. EAF Mass & Energy Balance!D100'
        nm3_O2_per_tLS = (
            (moles_O2_per_tLS * 8.314 * 273.15) / 101325
        ) * 1000  # Nm^3 O2 required per tLS, '12. EAF Mass & Energy Balance!D101'

        # Electric Arc Furnace (EAF) Fed with Scrap - Flux Addition
        CaO_MgO_ratio = 56.00 / 40.00  # (kg/kg), '12. EAF Mass & Energy Balance!D113'
        mass_MgO_doloma = output_dict["mass_MgO_slag_per_tLS"]  # (kg/tLS), '12. EAF Mass & Energy Balance!D114'
        mass_CaO_doloma = (
            mass_MgO_doloma * CaO_MgO_ratio
        )  # (kg/tLS), '12. EAF Mass & Energy Balance!D115'
        mass_doloma = (
            mass_MgO_doloma + mass_CaO_doloma
        )  # (kg/tLS), '12. EAF Mass & Energy Balance!D116'
        mass_lime = (
            mass_CaO_slag_per_tLS - mass_CaO_doloma
        )  # (kg/tLS), '12. EAF Mass & Energy Balance!D117'
        mass_doloma + mass_lime  # (kg/tLS), '12. EAF Mass & Energy Balance!D118'

        # Electric Arc Furnace (EAF) Fed with Scrap Only - Energy Balance
        # Inputs into EAF (feedstocks)
        # Scrap, Flux, Oxygen, Carbon
        # NOTE: Possibly replace these mole values with actual enthalpy calculations from excel sheet?
        scrap_Fe_J_mol = 0.0  # H (J/mol) Fe, '12. EAF Mass & Energy Balance!D124' > '14. Enthalpy Calculations!C113'
        scrap_Fe_kg = (
            output_dict["mass_scrap_per_tLS"] * scrap_composition["Fe"]
        )  # kg Fe, '12. EAF Mass & Energy Balance!F124'
        scrap_Fe_n_kmol = scrap_Fe_kg / 55.80  # kmol Fe, '12. EAF Mass & Energy Balance!E124'
        scrap_Fe_kJ = (
            scrap_Fe_J_mol * scrap_Fe_n_kmol
        )  # kJ Fe, '12. EAF Mass & Energy Balance!G124'

        scrap_SiO2_J_mol = (
            -9.0830e05
        )  # H (J/mol) SiO2, '12. EAF Mass & Energy Balance!D125' > '14. Enthalpy Calculations!C207'
        scrap_SiO2_kg = mass_SiO2_scrap_per_tLS  # kg SiO2, '12. EAF Mass & Energy Balance!F125' > '12. EAF Mass & Energy Balance!D71'
        scrap_SiO2_n_kmol = scrap_SiO2_kg / 60.00  # kmol SiO2, '12. EAF Mass & Energy Balance!E125'
        scrap_SiO2_kJ = (
            scrap_SiO2_J_mol * scrap_SiO2_n_kmol
        )  # kJ SiO2, '12. EAF Mass & Energy Balance!G125'

        flux_CaO_J_mol = (
            -6.3490e05
        )  # H (J/mol) CaO, '12. EAF Mass & Energy Balance!D126' > '14. Enthalpy Calculations!C93'
        flux_CaO_kg = mass_CaO_slag_per_tLS  # kg CaO, '12. EAF Mass & Energy Balance!F126' > '12. EAF Mass & Energy Balance!D76'
        flux_CaO_n_kmol = flux_CaO_kg / 56.0  # kmol CaO, '12. EAF Mass & Energy Balance!E126'
        flux_CaO_kJ = (
            flux_CaO_J_mol * flux_CaO_n_kmol
        )  # kJ CaO, '12. EAF Mass & Energy Balance!G126'

        flux_MgO_J_mol = (
            -6.0160e05
        )  # H (J/mol) MgO, '12. EAF Mass & Energy Balance!D127' > '14. Enthalpy Calculations!C181'
        flux_MgO_kg = output_dict["mass_MgO_slag_per_tLS"]  # kg MgO, '12. EAF Mass & Energy Balance!F127' > '12. EAF Mass & Energy Balance!D81'
        flux_MgO_n_kmol = flux_MgO_kg / 40.0  # kmol MgO, '12. EAF Mass & Energy Balance!E127'
        flux_MgO_kJ = (
            flux_MgO_J_mol * flux_MgO_n_kmol
        )  # kJ MgO, '12. EAF Mass & Energy Balance!G127'

        O2_J_mol = 0.0  # H (J/mol) O2, '12. EAF Mass & Energy Balance!D128' > '14. Enthalpy Calculations!C220'
        O2_n_kmol = moles_O2_per_tLS  # kmol O2, '12. EAF Mass & Energy Balance!E128' > '12. EAF Mass & Energy Balance!D100'
        O2_n_kmol * 32.0  # kg O2, '12. EAF Mass & Energy Balance!F128'
        O2_kJ = O2_J_mol * O2_n_kmol  # kJ O2, '12. EAF Mass & Energy Balance!G128'

        C_J_mol = 0.0  # H (J/mol) C from NG and injected Carbon, '12. EAF Mass & Energy Balance!D129' > '14. Enthalpy Calculations!C69'
        C_kg = (
            mass_C_ng_per_tLS + mass_injected_carbon_per_tLS
        )  # kg C from NG and injected Carbon, '12. EAF Mass & Energy Balance!F129'
        C_n_kmol = (
            C_kg / 12.0
        )  # kmol C from NG and injected Carbon, '12. EAF Mass & Energy Balance!E129' > '12. EAF Mass & Energy Balance!D241'
        C_kJ = (
            C_J_mol * C_n_kmol
        )  # kJ C from NG and injected Carbon, '12. EAF Mass & Energy Balance!G129'

        total_EAF_scrap_inputs_kJ = (
            scrap_Fe_kJ + scrap_SiO2_kJ + flux_CaO_kJ + flux_MgO_kJ + O2_kJ + C_kJ
        )  # total kJ EAF inputs, '12. EAF Mass & Energy Balance!G130'

        # EAF Products
        # Steel, Slag, Off-gas
        # NOTE: Possibly replace these mole values with actual enthalpy calculations from excel sheet?
        steel_Fe_J_mol = 7.583849597377010e04  # H (J/mol) Fe in Steel product, '12. EAF Mass & Energy Balance!D134' > '14. Enthalpy Calculations!C364'
        steel_Fe_kg = mass_iron_per_tLS  # kg Fe in Steel product, '12. EAF Mass & Energy Balance!F134' > '12. EAF Mass & Energy Balance!D7'
        steel_Fe_kmol = (
            steel_Fe_kg / 55.80
        )  # kmol Fe in Steel product, '12. EAF Mass & Energy Balance!E134'
        steel_Fe_kJ = (
            steel_Fe_J_mol * steel_Fe_kmol
        )  # kJ Fe in Steel product, '12. EAF Mass & Energy Balance!G134'

        steel_C_J_mol = 3.220145150706920e04  # H (J/mol) C in Steel product, '12. EAF Mass & Energy Balance!D135' > '14. Enthalpy Calculations!C371'
        steel_C_kg = mass_carbon_per_tLS  # kg C in Steel product, '12. EAF Mass & Energy Balance!F135' > '12. EAF Mass & Energy Balance!D8'
        steel_C_kmol = (
            steel_C_kg / 12.0
        )  # kmol C in Steel product, '12. EAF Mass & Energy Balance!E135'
        steel_C_kJ = (
            steel_C_J_mol * steel_C_kmol
        )  # kJ C in Steel product, '12. EAF Mass & Energy Balance!G135'

        slag_FeO_kg = output_dict["mass_FeO_slag_per_tLS"]  # kg FeO in slag product, '12. EAF Mass & Energy Balance!F136' > '12. EAF Mass & Energy Balance!D82'
        slag_SiO2_kg = mass_SiO2_slag_per_tLS  # kg SiO2 in slag product, '12. EAF Mass & Energy Balance!F137' > '12. EAF Mass & Energy Balance!D74'
        slag_Al2O3_kg = mass_Al2O3_slag_per_tLS  # kg Al2O3 in slag product, '12. EAF Mass & Energy Balance!F138' > '12. EAF Mass & Energy Balance!D75'
        slag_CaO_kg = mass_CaO_slag_per_tLS  # kg CaO in slag product, '12. EAF Mass & Energy Balance!F139' > '12. EAF Mass & Energy Balance!D76'
        slag_MgO_kg = output_dict["mass_MgO_slag_per_tLS"]  # kg MgO in slag product, '12. EAF Mass & Energy Balance!F140' > '12. EAF Mass & Energy Balance!D81'
        slag_total_kJ = (
            -8.354013744345140e00
            * (slag_FeO_kg + slag_SiO2_kg + slag_Al2O3_kg + slag_CaO_kg + slag_MgO_kg)
        ) * 1000

        off_gas_CO_J_mol = -5.887443663594190e04  # H (J/mol) CO in off-gas product, '12. EAF Mass & Energy Balance!D141' > '14. Enthalpy Calculations!C322'
        off_gas_CO_kmol = moles_CO_injected_per_tLS  # kmol CO in off-gas product, '12. EAF Mass & Energy Balance!E141' > '12. EAF Mass & Energy Balance!D98'
        off_gas_CO_kJ = (
            off_gas_CO_J_mol * off_gas_CO_kmol
        )  # kJ CO in off-gas product, '12. EAF Mass & Energy Balance!G141'

        off_gas_CO2_J_mol = -3.10933969795530e05  # H (J/mol) CO2 in off-gas product, '12. EAF Mass & Energy Balance!D142' > '14. Enthalpy Calculations!C327'
        off_gas_CO2_kmol = moles_CO2_ng_per_tLS  # kmol CO2 in off-gas product, '12. EAF Mass & Energy Balance!E142' > '12. EAF Mass & Energy Balance!D94'
        off_gas_CO2_kJ = (
            off_gas_CO2_J_mol * off_gas_CO2_kmol
        )  # kJ CO2 in off-gas product, '12. EAF Mass & Energy Balance!G142'

        total_EAF_scrap_products_kJ = (
            steel_Fe_kJ + steel_C_kJ + slag_total_kJ + off_gas_CO_kJ + off_gas_CO2_kJ
        )  # total kJ EAF products, '12. EAF Mass & Energy Balance!G143'

        EAF_scrap_energy_consumption_kJ_tHM = (
            total_EAF_scrap_products_kJ - total_EAF_scrap_inputs_kJ
        )  # (kJ/tHM) EAF scrap energy consumption, '12. EAF Mass & Energy Balance!G145'
        EAF_scrap_energy_consumption_kWh_tHM = (
            EAF_scrap_energy_consumption_kJ_tHM / kWh_to_kJ
        )  # (kWh/tHM) EAF scrap energy consumption, '12. EAF Mass & Energy Balance!G146'

        # NOTE: 470.00 is a hardcoded value / assumption input on '5. Electric Arc Furnace'!C6
        EAF_scrap_heat_loss_adjustment_abs = (
            470.00 - EAF_scrap_energy_consumption_kWh_tHM
        )  # (kWh/tHM) EAF scrap absolute heat loss adjustment, '12. EAF Mass & Energy Balance!G148'
        (
            EAF_scrap_heat_loss_adjustment_abs / 470.00
        )  # % EAF scrap heat loss adjustment, '12. EAF Mass & Energy Balance!G149'

        EAF_scrap_energy_consumption_w_heat_loss_kWh_tHM = (
            EAF_scrap_energy_consumption_kWh_tHM + EAF_scrap_heat_loss_adjustment_abs
        )  # (kWh/tHM) Total EAF with scrap energy consumption with heat loss adjustment, '12. EAF Mass & Energy Balance!G151'

        # 5. Electric Arc Furnace > Scrap Only
        # Electric Arc Furnace (EAF) with Scrap process Inputs (Feedstocks)
        output_dict["oxygen_per_tLS"] = nm3_O2_per_tLS  # Nm^3 O2/ton Steel, '12. EAF Mass & Energy Balance!D101' > '5. Electric Arc Furnace!C5'
        output_dict["electricity_per_tLS"] = EAF_scrap_energy_consumption_w_heat_loss_kWh_tHM  # kWh/ton Hot Metal / Liquid Steel?, Harcoded as 470, '12. EAF Mass & Energy Balance!G151' >'5. Electric Arc Furnace!C6'
        output_dict["natural_gas_per_tLS"] = (
            natural_gas  # MMBtu/ton steel, Hardocded as 0.44, '5. Electric Arc Furnace!C7'
        )
        # electrodes = electrodes  # kg/ton steel,  Hardocded as 2.0, '5. Electric Arc Furnace!C8'
        output_dict["scrap_steel_per_tLS"] = (
            output_dict["mass_scrap_per_tLS"] / 1000
        )  # ton, '5. Electric Arc Furnace!C9' > '12. EAF Mass & Energy Balance!D27/1000'
        output_dict["coal_per_tLS"] = (
            mass_injected_carbon_per_tLS / 0.806 / 1000
        )  # ton, assum 0.806 tonC/tonCoal, '5. Electric Arc Furnace!C10' > '12. EAF Mass & Energy Balance!D91/0.806/1000'
        output_dict["burnt_doloma_per_tLS"] = (
            mass_doloma / 1000
        )  # ton, '5. Electric Arc Furnace!C11' > '12. EAF Mass & Energy Balance!D116/1000'
        output_dict["burnt_lime_per_tLS"] = (
            mass_lime / 1000
        )  # ton, '5. Electric Arc Furnace!C12' > '12. EAF Mass & Energy Balance!D117/1000'

        return output_dict

class ElectricArcFurnaceDRIPerformanceComponent(PerformanceModelBaseClass):
    def initialize(self):
        super.initialize()
        # NOTE: do we need to setup self.commodity* here?

    def setup(self):
        super.setup()

        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]

        # NOTE: likely need to update this base config class or create a new one with appropriate inputs?
        self.config = ElectricArcFurnacePerformanceBaseConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=True,
            additional_cls_name=self.__class__.__name__,
        )

        # NOTE: update based on feedback on baseconfig class
        # steel_capacity = mTons/year, 'Model Inputs & Outputs!B11'
        self.add_input(
            "system_capacity",
            val=self.config.steel_production_rate_tonnes_per_hr,
            units=self.commodity_rate_units,
            desc="Rated steel production capacity",
        )

        # annual_production = 2.0 mTons/year, 'Model Inputs & Outputs!B12'
        self.add_input(
            "annual_production",
            val=self.config.steel_production_rate_tonnes_per_hr,  # update
            units=self.commodity_rate_units,
            desc="Rated steel production capacity",
        )

        # NOTE: was going to add plant life = 40 years here but that seems to be part of the cost component in other models?

        # NOTE: update with proper feedstocks (everything under "Input feedstocks for EAF Fed with DRI Directly (No ESF)")
        # NOTE: oxygen, electricity, natural gas, electrodes, scrap steel, coal, burnt doloma, burnt lime
        # Add feedstock inputs and outputs, default to 0 --> set using feedstock component
        # Add feedstock inputs and outputs, default to 0 --> set using feedstock component
        for feedstock, feedstock_units in self.feedstocks_to_units.items():
            self.add_input(
                f"{feedstock}_in",
                val=0.0,
                shape=n_timesteps,
                units=feedstock_units,
                desc=f"{feedstock} available for steel production",
            )
            self.add_output(
                f"{feedstock}_consumed",
                val=0.0,
                shape=n_timesteps,
                units=feedstock_units,
                desc=f"{feedstock} consumed for steel production",
            )

        # NOTE, not sure where this distinction of pellet_grade is made up stream but replace this with that (from config?)

        # Add outputs
        # NOTE: need clarity on units and how to represent them here
        self.add_output(
            "mass slag",  # == mass_slag_per_tDRI
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # Total kg slag per tDRI
            desc="Total kg slag per ton DRI",
        )

        self.add_output(
            "mass MgO slag",  # == mass_MgO_slag_per_tDRI
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg MgO in slag per ton DRI
            desc="Total kg MgO in slag per ton DRI",
        )

        self.add_output(
            "mass FeO slag",  # == mass_FeO_slag_per_tDRI
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg FeO in slag per ton DRI
            desc="Total kg FeO in slag per ton DRI",
        )

        self.add_output(
            "mass FeO DRI",  # == mass_FeO_DRI_per_tDRI
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg FeO from DRI per ton DRI
            desc="Total kg FeO from DRI per ton DRI",
        )

        self.add_output(
            "added mass FeO for slag",  # == add_mass_FeO_needed
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg additional FeO need for slag per ton DRI
            desc="Total additional kg FeO need for slag per ton DRI",
        )

        self.add_output(
            "mass Fe to FeO",  # == mass_Fe_to_FeO
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg Fe consumed to produce FeO per ton DRI
            desc="Total kg Fe consumed to produce FeO per ton DRI",
        )

        self.add_output(
            "mass Fe from DRI",  # == mass_Fe_DRI_per_tDRI
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg Fe from DRI per ton DRI
            desc="Total kg Fe from DRI per ton DRI",
        )

        self.add_output(
            "mass Fe from scrap",  # == mass_Fe_scrap_per_tDRI
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg Fe from scrap per ton DRI
            desc="Total kg Fe from scrap per ton DRI",
        )

        self.add_output(
            "mass steel",  # == mass_steel_per_tDRI
            val=0.0,
            shape=n_timesteps,  # NOTE: currently single value, need to vectorize for timeseries calcs
            # units= # kg Steel formed from DRI + scrap per ton DRI
            desc="Total kg Steel formed from DRI + scrap per ton DRI",
        )

    def compute(self, inputs, outputs):
        # NOTE: where should these live as inputs / config / feedstocks?
        # Steel and scrap composition
        pct_carbon_steel = 0.1 / 100  # mass fraction C, 'Model Inputs & Outputs!B26'
        scrap_composition = {
            "Fe": 94.0 / 100,  # mass fraction Fe, 'Model Inputs & Outputs!B27'
            "SiO2": 1.0 / 100,  # mass fraction SiO2, 'Model Inputs & Outputs!B28'
        }

        # NOTE: these DRI values are calculated values from 10. DRI Mass & Energy Balance
        # NOTE: replace with composition of DRI coming out of LBNL model
        if pellet_grade == "BF":
            DRI_composition = {
                "Fe": 0.8019049064951670,  # mass fraction Fe, '10. DRI Mass & Energy Balance'!D73 > '12. EAF Mass & Energy Balance!D13'
                "FeO": 0.06586224237743430,  # mass fraction FeO, '10. DRI Mass & Energy Balance'!D74 > '12. EAF Mass & Energy Balance!D14'
                "gangue": 0.112232851127399000,  # mass fraction gangue, '10. DRI Mass & Energy Balance'!D75 > '12. EAF Mass & Energy Balance!D15'
                "C": 0.020,  # mass fraction C, '10. DRI Mass & Energy Balance'!D76 > '12. EAF Mass & Energy Balance!D16'
            }
            SiO2_ratio = (
                3.0  # kg/kg SiO2 to Alumina Ratio in DRI, '12. EAF Mass & Energy Balance!D162'
            )

        if pellet_grade == "DR":
            DRI_composition = {
                "Fe": 0.8431916497235140,  # mass fraction Fe, '10. DRI Mass & Energy Balance'!E73 > '12. EAF Mass & Energy Balance!E13'
                "FeO": 0.06925321488234770,  # mass fraction FeO, '10. DRI Mass & Energy Balance'!E74 > '12. EAF Mass & Energy Balance!E14'
                "gangue": 0.06755513539413880,  # mass fraction gangue, '10. DRI Mass & Energy Balance'!E75 > '12. EAF Mass & Energy Balance!E15'
                "C": 0.020,  # mass fraction C, '10. DRI Mass & Energy Balance'!E76 > '12. EAF Mass & Energy Balance!E16'
            }

            SiO2_ratio = (
                1.25  # kg/kg SiO2 to Alumina Ratio in DRI, '12. EAF Mass & Energy Balance!E162'
            )

        # NOTE update below 3 feedstock sections to get feedstocks appropriately based on updates in setup
        # EAF operation
        pct_DRI = 60.0 / 100  # mass fraction, 'Model Inputs & Outputs!B61'
        DRI_feed_temp = 873  # hot = 873 K or cold = 298 K, 'Model Inputs & Outputs!B63'

        # get feedstocks

        # get feedstock usage rates
        # Including DRI in feed (assumed constants in feedstocks)
        natural_gas = 0.44  # MMBtu/ton steel, '5. Electric Arc Furnace!C32'
        electrodes = 2.00  # kg/ton steel, '5. Electric Arc Furnace!C33'

        # 12. EAF Mass & Energy Balance
        # Essential Mass Summary
        mass_steel_stream = 1000.0  # kg liquid steel, '12. EAF Mass & Energy Balance!D4'
        pct_carbon_steel = pct_carbon_steel  # % mass C, 'Model Inputs & Outputs!B26' > '12. EAF Mass & Energy Balance!D5'
        mass_iron_per_tLS = mass_steel_stream * (
            1 - pct_carbon_steel
        )  # kg Fe/ton liquid steel, '12. EAF Mass & Energy Balance!D7'
        mass_carbon_per_tLS = (
            mass_steel_stream * pct_carbon_steel
        )  # kg C/ton liquid steel, '12. EAF Mass & Energy Balance!D8'
        scrap_composition = scrap_composition  # % mass Fe, 'Model Inputs & Outputs!B27'
        # % mass SiO2, 'Model Inputs & Outputs!B28'

        # Essential Mass Summary > Burden of Composition DRI-EAF
        # NOTE: calculated relative to 1 ton DRI (tDRI)
        share_of_DRI_in_charge = (
            pct_DRI  # mass %, 'Model Inputs & Outputs!B61' > '12. EAF Mass & Energy Balance!D29'
        )

        # Electric Arc Furnace Fed with DRI Directly (No ESF) - Mass Balance (Fe, C, O, MgO, SiO2, Al2O3, CaO)
        # NOTE: calculated per ton DRI (tDRI)
        mass_basis_DRI = 1000  # kg, '12. EAF Mass & Energy Balance!D158'
        mass_scrap_from_basis = (
            mass_basis_DRI - share_of_DRI_in_charge * mass_basis_DRI
        ) / share_of_DRI_in_charge  # kg, '12. EAF Mass & Energy Balance!D159'

        mass_gangue_per_tDRI = (
            mass_basis_DRI * DRI_composition["gangue"]
        )  # kg gangue/tDRI, '12. EAF Mass & Energy Balance!D161'
        SiO2_ratio = (
            SiO2_ratio  # kg/kg SiO2 to Alumina Ratio in DRI, '12. EAF Mass & Energy Balance!D162'
        )
        mass_SiO2_DRI_per_tDRI = (mass_gangue_per_tDRI * SiO2_ratio) / (
            SiO2_ratio + 1
        )  # kg SiO2/tDRI, '12. EAF Mass & Energy Balance!D163'
        mass_Al2O3_DRI_per_tDRI = mass_gangue_per_tDRI / (
            SiO2_ratio + 1
        )  # kg Al2O3/tDRI, '12. EAF Mass & Energy Balance!D164'

        mass_pct_SiO2_scrap = scrap_composition[
            "SiO2"
        ]  # % mass SiO2, 'Model Inputs & Outputs!B28' > '12. EAF Mass & Energy Balance!D166'
        mass_SiO2_scrap_per_tDRI = (
            mass_pct_SiO2_scrap * mass_scrap_from_basis
        )  # kg SiO2/tDRI '12. EAF Mass & Energy Balance!D167'

        slag_B3 = 1.50  # basicity, kg CaO / (kg SiO2 + kg Al2O3) # kg SiO2/tDRI, '12. EAF Mass & Energy Balance!D169'
        mass_CaO_per_tDRI = slag_B3 * (
            mass_SiO2_scrap_per_tDRI + mass_Al2O3_DRI_per_tDRI + mass_SiO2_DRI_per_tDRI
        )  # kg CaO mass added to EAF per tDRI, '12. EAF Mass & Energy Balance!D170'

        mass_SiO2_slag_per_tDRI = (
            mass_SiO2_DRI_per_tDRI + mass_SiO2_scrap_per_tDRI
        )  # kg SiO2/tDRI, '12. EAF Mass & Energy Balance!D172'
        mass_Al2O3_slag_per_tDRI = (
            mass_Al2O3_DRI_per_tDRI  # kg AlO3/tDRI, '12. EAF Mass & Energy Balance!D173'
        )
        mass_CaO_slag_per_tDRI = (
            mass_CaO_per_tDRI  # kg CaO/tDRI, '12. EAF Mass & Energy Balance!D174'
        )

        pct_MgO_slag = (
            12.0 / 100
        )  # mass fraction MgO of slag, assumed input, '12. EAF Mass & Energy Balance!D176'
        pct_FeO_slag = (
            30.0 / 100
        )  # mass fraction FeO of slag, assumed input, '12. EAF Mass & Energy Balance!D177'
        mass_slag_per_tDRI = (
            mass_SiO2_slag_per_tDRI + mass_Al2O3_slag_per_tDRI + mass_CaO_slag_per_tDRI
        ) / (1 - pct_FeO_slag - pct_MgO_slag)  # kg slag/tDRI, '12. EAF Mass & Energy Balance!D178'
        mass_MgO_slag_per_tDRI = (
            pct_MgO_slag * mass_slag_per_tDRI
        )  # kg MgO in slag/tDRI, '12. EAF Mass & Energy Balance!D179'
        mass_FeO_slag_per_tDRI = (
            pct_FeO_slag * mass_slag_per_tDRI
        )  # kg FeO in slag/tDRI, '12. EAF Mass & Energy Balance!D180'

        mass_FeO_DRI_per_tDRI = (
            mass_basis_DRI * DRI_composition["FeO"]
        )  # kg FeO/tDRI, '12. EAF Mass & Energy Balance!D182'
        (
            mass_FeO_DRI_per_tDRI * 71.80
        )  # kmol FeO/tDRI, 71.80 = '10. DRI Mass & Energy Balance!D22', '12. EAF Mass & Energy Balance!D183'
        add_mass_FeO_needed = (
            mass_FeO_slag_per_tDRI - mass_FeO_DRI_per_tDRI
        )  # kg additional mass FeO required for slag per tDRI, '12. EAF Mass & Energy Balance!D184'
        add_moles_FeO_needed = (
            add_mass_FeO_needed / 71.80
        )  # kmol additional mass FeO required for slag per tDRI, '12. EAF Mass & Energy Balance!D185'
        moles_Fe_to_FeO = add_moles_FeO_needed  # mole Fe consumed to produce FeO per tDRI, '12. EAF Mass & Energy Balance!D186'
        mass_Fe_to_FeO = (
            moles_Fe_to_FeO * 55.80
        )  # kg Fe consumed to produce FeO per tDRI, '12. EAF Mass & Energy Balance!D187'

        mass_Fe_DRI_per_tDRI = (
            mass_basis_DRI * DRI_composition["Fe"] - mass_Fe_to_FeO
        )  # kg Fe from DRI per tDRI, '12. EAF Mass & Energy Balance!D189'
        mass_Fe_scrap_per_tDRI = (
            mass_scrap_from_basis * scrap_composition["Fe"]
        )  # kg Fe from scrap per tDRI, '12. EAF Mass & Energy Balance!D190'
        mass_Fe_per_tDRI = (
            mass_Fe_DRI_per_tDRI + mass_Fe_scrap_per_tDRI
        )  # kg Fe from DRI + scrap per tDRI, '12. EAF Mass & Energy Balance!D191'
        mass_steel_per_tDRI = mass_Fe_per_tDRI / (
            1 - pct_carbon_steel
        )  # kg Steel formed from DRI + scrap per tDRI, '12. EAF Mass & Energy Balance!D192'

        # NOTE: calculated per ton liquid steel (tLS)
        mass_DRI_per_tLS = (
            mass_basis_DRI / mass_steel_per_tDRI
        ) * 1000  # kg DRI per tLS, '12. EAF Mass & Energy Balance!D195'
        mass_scrap_per_tLS = (
            mass_scrap_from_basis / mass_steel_per_tDRI
        ) * 1000  # kg scrap per tLS, '12. EAF Mass & Energy Balance!D196'

        mass_gangue_per_tLS = (
            mass_DRI_per_tLS * DRI_composition["gangue"]
        )  # kg gangue per tLS from DRI, '12. EAF Mass & Energy Balance!D198'
        SiO2_ratio = SiO2_ratio  # kg/kg SiO2 to Alumina Ratio in DRI, '12. EAF Mass & Energy Balance!D199' > '12. EAF Mass & Energy Balance!D162'
        mass_SiO2_DRI_per_tLS = (mass_gangue_per_tLS * SiO2_ratio) / (
            SiO2_ratio + 1
        )  # kg SiO2 per tLS from DRI, '12. EAF Mass & Energy Balance!D200'
        mass_SiO2_scrap_per_tLS = (
            mass_scrap_per_tLS * mass_pct_SiO2_scrap
        )  # kg SiO2 per tLS from scrap, '12. EAF Mass & Energy Balance!D201'
        mass_Al2O3_per_tLS = mass_gangue_per_tLS / (
            SiO2_ratio + 1
        )  # kg Al2O3 per tLS from DRI, '12. EAF Mass & Energy Balance!D202'

        slag_B3 = slag_B3  # basicity, kg CaO / (kg SiO2 + kg Al2O3), '12. EAF Mass & Energy Balance!D204' > '12. EAF Mass & Energy Balance!D169'
        mass_SiO2_slag_per_tLS = (
            mass_SiO2_DRI_per_tLS + mass_SiO2_scrap_per_tLS
        )  # kg SiO2 in slag per tLS, '12. EAF Mass & Energy Balance!D205'
        mass_Al2O3_slag_per_tLS = mass_Al2O3_per_tLS  # kg Al2O3 in slag per tLS, '12. EAF Mass & Energy Balance!D206' > '12. EAF Mass & Energy Balance!D202'
        mass_CaO_slag_per_tLS = slag_B3 * (
            mass_SiO2_slag_per_tLS + mass_Al2O3_slag_per_tLS
        )  # kg CaO in slag per tLS, '12. EAF Mass & Energy Balance!D207'

        pct_MgO_slag = pct_MgO_slag  # mass fraction MgO in slag, assumed input, '12. EAF Mass & Energy Balance!D209' > '12. EAF Mass & Energy Balance!D176'
        pct_FeO_slag = pct_FeO_slag  # mass fraction FeO in slag, assumed input, '12. EAF Mass & Energy Balance!D210' > '12. EAF Mass & Energy Balance!D177'
        mass_slag_per_tLS = (
            mass_SiO2_slag_per_tLS + mass_Al2O3_slag_per_tLS + mass_CaO_slag_per_tLS
        ) / (
            1 - pct_MgO_slag - pct_FeO_slag
        )  # kg slag per tLS, '12. EAF Mass & Energy Balance!D211'
        mass_MgO_slag_per_tLS = (
            pct_MgO_slag * mass_slag_per_tLS
        )  # kg MgO in slag per tLS, '12. EAF Mass & Energy Balance!D212'
        mass_FeO_slag_per_tLS = (
            pct_FeO_slag * mass_slag_per_tLS
        )  # kg FeO in slag per tLS, '12. EAF Mass & Energy Balance!D213'

        mass_FeO_DRI_per_tLS = (
            mass_DRI_per_tLS * DRI_composition["FeO"]
        )  # kg FeO from DRI per tLS, '12. EAF Mass & Energy Balance!D215'
        mole_FeO_DRI_per_tLS = (
            mass_FeO_DRI_per_tLS / 71.8
        )  # kmol FeO from DRI per tLS, '12. EAF Mass & Energy Balance!D216'
        add_mass_FeO_needed_tLS = (
            mass_FeO_slag_per_tLS - mass_FeO_DRI_per_tLS
        )  # kg additional FeO required from slag per tLS, '12. EAF Mass & Energy Balance!D217'
        add_moles_FeO_needed_tLS = (
            add_mass_FeO_needed_tLS / 71.8
        )  # kmol additional FeO required from slag per tLS, '12. EAF Mass & Energy Balance!D218'
        moles_Fe_to_FeO_tLS = add_moles_FeO_needed_tLS  # kmol Fe consumed to produce FeO per tLS, '12. EAF Mass & Energy Balance!D219'
        (
            moles_Fe_to_FeO_tLS * 55.80
        )  # kg Fe consumed to produce FeO per tLS, '12. EAF Mass & Energy Balance!D220'
        moles_O2_to_FeO_tLS = (
            moles_Fe_to_FeO_tLS * 0.5
        )  # kmol O2 consumed to produce FeO per tLS, '12. EAF Mass & Energy Balance!D221'

        mass_C_steel_per_tLS = (
            mass_steel_stream * pct_carbon_steel
        )  # kg Carbon in Steel per tLS, '12. EAF Mass & Energy Balance!D223'
        mass_C_DRI_per_tLS = (
            mass_DRI_per_tLS * DRI_composition["C"]
        )  # kg Carbon in DRI per tLS, '12. EAF Mass & Energy Balance!D224'
        mass_C_ng_per_tLS = (
            natural_gas * MMbtu_to_MJ / 50.0 * 12.00 / 16.00
        )  # kg Carbon in natural gas per tLS, '12. EAF Mass & Energy Balance!D225'
        # NOTE: 50.0 = LHV CH4, 12 = molar mass C, 16.00 = molar mass CH4
        pct_carbon_steel_tap = (
            3 / 100
        )  # mass fraction carbon input to EAF as % of steel tap mass, '12. EAF Mass & Energy Balance!D226'
        total_C_kg_per_tLS = (
            pct_carbon_steel_tap * 1000
        )  # kg total Carbon in put per tLS, '12. EAF Mass & Energy Balance!D227'
        if (
            total_C_kg_per_tLS - mass_C_DRI_per_tLS
        ) > 0:  # kg additiona Carbon required per tLS, '12. EAF Mass & Energy Balance!D228'
            mass_injected_carbon_per_tLS = total_C_kg_per_tLS - mass_C_DRI_per_tLS
        else:
            mass_injected_carbon_per_tLS = 0
        moles_C_ng_per_tLS = (
            mass_C_ng_per_tLS / 12.0
        )  # kmol Carbon in NG blown out per tLS, '12. EAF Mass & Energy Balance!D229'
        moles_O2_ng_per_tLS = (
            moles_C_ng_per_tLS * 1
        )  # kmol Oxygen needed to blow out NG per tLS, '12. EAF Mass & Energy Balance!D230', carbon in NG oxidizes to CO2 immediately
        moles_CO2_ng_per_tLS = (
            moles_C_ng_per_tLS * 1
        )  # kmol CO2 formed from NG, '12. EAF Mass & Energy Balance!D231'
        (moles_CO2_ng_per_tLS * 44.0)  # kg CO2 formed from NG, '12. EAF Mass & Energy Balance!D232'
        moles_C_DRI_per_tLS = (
            (mass_C_DRI_per_tLS - mass_C_steel_per_tLS) / 12.00
        )  # kmol Carbon in DRI blown out per tLS, '12. EAF Mass & Energy Balance!D233', assume remaining C originated in DRI
        moles_O2_DRI_per_tLS = (
            moles_C_DRI_per_tLS * 0.5
        )  # kmol Oxygen needed to blow out C in DRI per tLS, '12. EAF Mass & Energy Balance!D234'
        moles_CO_DRI_per_tLS = moles_C_DRI_per_tLS  # kmol CO formmed from C in DRI per tLS, '12. EAF Mass & Energy Balance!D235'
        mass_CO_DRI_per_tLS = (
            moles_CO_DRI_per_tLS * 28.0
        )  # kg CO formed from C in DRI per tLS, '12. EAF Mass & Energy Balance!D236'
        moles_C_injected_per_tLS = (
            mass_injected_carbon_per_tLS / 12.0
        )  # kmol C injected carbon blown out per tLS, '12. EAF Mass & Energy Balance!D237', assume remaining C in steel originated in DRI or injected carbon
        moles_O2_injected_per_tLS = (
            moles_C_injected_per_tLS * 0.5
        )  # kmol O2 needed to blow out C in injected carbon, '12. EAF Mass & Energy Balance!D238'
        moles_CO_injected_per_tLS = (
            moles_C_injected_per_tLS * 1
        )  # kmol CO formed from C in injected carbon, '12. EAF Mass & Energy Balance!D239'
        mass_CO_injected_per_tLS = (
            moles_CO_injected_per_tLS * 28.0
        )  # kg CO formed from C in injected carbon, '12. EAF Mass & Energy Balance!D240'
        moles_O2_per_tLS = (
            moles_O2_ng_per_tLS
            + moles_O2_injected_per_tLS
            + moles_O2_to_FeO_tLS
            + moles_O2_DRI_per_tLS
        )  # kmol O2 required per tLS, '12. EAF Mass & Energy Balance!D241'
        nm3_O2_per_tLS = (
            ((moles_O2_per_tLS * 8.314 * 273.15) / 101325) * 1000
        )  # Nm^3 O2 required per tLS, '12. EAF Mass & Energy Balance!D242' (ideal gas situtation)

        # Electric Arc Furnace (EAF) Fed with DRI Directly (No ESF) - Flux Addition
        CaO_MgO_ratio = 56.00 / 40.00  # (kg/kg), '12. EAF Mass & Energy Balance!D254'
        mass_MgO_doloma = mass_MgO_slag_per_tLS  # (kg/tLS), '12. EAF Mass & Energy Balance!D255'
        mass_CaO_doloma = (
            mass_MgO_doloma * CaO_MgO_ratio
        )  # (kg/tLS), '12. EAF Mass & Energy Balance!D256'
        mass_doloma = (
            mass_MgO_doloma + mass_CaO_doloma
        )  # (kg/tLS), '12. EAF Mass & Energy Balance!D257'
        mass_lime = (
            mass_CaO_slag_per_tLS - mass_CaO_doloma
        )  # (kg/tLS), '12. EAF Mass & Energy Balance!D258'
        mass_doloma + mass_lime  # (kg/tLS), '12. EAF Mass & Energy Balance!D259'

        # Electric Arc Furnace (EAF) Fed with DRI Directly (No ESF) - Energy Balance
        # Inputs into EAF (feedstocks)
        # DRI, Scrap, Flux, Oxygen, Carbon
        # NOTE: Possibly replace these mole values with actual enthalpy calculations from excel sheet?
        if DRI_feed_temp == 873:
            DRI_Fe_J_mol = 1.8432477097027300e04  # H (J/mol) Fe, '12. EAF Mass & Energy Balance!D264' > '14. Enthalpy Calculations!C235'
            DRI_FeO_J_mol = -2.346170978905830e05  # H (J/mol) FeO, '12. EAF Mass & Energy Balance!D265' > '14. Enthalpy Calculations!C272'
            DRI_C_J_mol = 9.144557831628680e03  # H (J/mol) C, '12. EAF Mass & Energy Balance!D266' > '14. Enthalpy Calculations!C286'
            DRI_SiO2_J_mol = -8.724350519581140e05  # H (J/mol) SiO2, '12. EAF Mass & Energy Balance!D267' > '14. Enthalpy Calculations!C281'
            DRI_Al2O3_J_mol = -1.613427222924770e06  # H (J/mol) Al2O3, '12. EAF Mass & Energy Balance!D268' > '14. Enthalpy Calculations!C232'

        if DRI_feed_temp == 298:
            DRI_Fe_J_mol = 0.0  # H (J/mol) Fe, '12. EAF Mass & Energy Balance!D264' > '14. Enthalpy Calculations!C113'
            DRI_FeO_J_mol = -2.65832239120e05  # H (J/mol) FeO, '12. EAF Mass & Energy Balance!D265' > '14. Enthalpy Calculations!C150'
            DRI_C_J_mol = 0.0  # H (J/mol) C, '12. EAF Mass & Energy Balance!D266' > '14. Enthalpy Calculations!C69'
            DRI_SiO2_J_mol = -9.0830e05  # H (J/mol) SiO2, '12. EAF Mass & Energy Balance!D267' > '14. Enthalpy Calculations!C207'
            DRI_Al2O3_J_mol = -1.675711853668660e06  # H (J/mol) Al2O3, '12. EAF Mass & Energy Balance!D268' > '14. Enthalpy Calculations!C56'

        DRI_Fe_kg = (
            mass_DRI_per_tLS * DRI_composition["Fe"]
        )  # kg Fe BF pellets, '12. EAF Mass & Energy Balance!F264'
        DRI_Fe_n_kmol = (
            DRI_Fe_kg / 55.80
        )  # kmol Fe BF pellets, '12. EAF Mass & Energy Balance!E264'
        DRI_Fe_kJ = (
            DRI_Fe_J_mol * DRI_Fe_n_kmol
        )  # kJ Fe BF pellets, '12. EAF Mass & Energy Balance!G264'

        DRI_FeO_J_mol = (
            -2.346170978905830e05
        )  # H (J/mol) FeO, '12. EAF Mass & Energy Balance!D265' > '14. Enthalpy Calculations!C272'
        DRI_FeO_n_kmol = (
            mole_FeO_DRI_per_tLS  # kmol FeO BF pellets, '12. EAF Mass & Energy Balance!E265'
        )
        DRI_FeO_kJ = (
            DRI_FeO_J_mol * DRI_FeO_n_kmol
        )  # kJ FeO BF pellets, '12. EAF Mass & Energy Balance!G265'

        DRI_C_J_mol = 9.144557831628680e03  # H (J/mol) C, '12. EAF Mass & Energy Balance!D266' > '14. Enthalpy Calculations!C286'
        DRI_C_kg = mass_C_DRI_per_tLS  # kg C BF pellets, '12. EAF Mass & Energy Balance!F266' > '12. EAF Mass & Energy Balance!D224'
        DRI_C_n_kmol = DRI_C_kg / 12.0  # kmol C BF pellets, '12. EAF Mass & Energy Balance!E266'
        DRI_C_kJ = (
            DRI_C_J_mol * DRI_C_n_kmol
        )  # kJ C BF pellets, '12. EAF Mass & Energy Balance!G266'

        DRI_SiO2_J_mol = (
            -8.724350519581140e05
        )  # H (J/mol) SiO2, '12. EAF Mass & Energy Balance!D267' > '14. Enthalpy Calculations!C281'
        DRI_SiO2_kg = mass_SiO2_DRI_per_tLS  # kg SiO2 BF pellets, '12. EAF Mass & Energy Balance!F267' > '12. EAF Mass & Energy Balance!D200'
        DRI_SiO2_n_kmol = (
            DRI_SiO2_kg / 60.0
        )  # kmol SiO2 BF pellets, '12. EAF Mass & Energy Balance!E267'
        DRI_SiO2_kJ = (
            DRI_SiO2_J_mol * DRI_SiO2_n_kmol
        )  # kJ SiO2 BF pellets, '12. EAF Mass & Energy Balance!G267'

        DRI_Al2O3_J_mol = -1.613427222924770e06  # H (J/mol) Al2O3, '12. EAF Mass & Energy Balance!D268' > '14. Enthalpy Calculations!C281'
        DRI_Al2O3_kg = mass_Al2O3_per_tLS  # kg Al2O3 BF pellets, '12. EAF Mass & Energy Balance!F268' > '12. EAF Mass & Energy Balance!D202'
        DRI_Al2O3_n_kmol = (
            DRI_Al2O3_kg / 102.0
        )  # kmol Al2O3 BF pellets, '12. EAF Mass & Energy Balance!E268'
        DRI_Al2O3_kJ = (
            DRI_Al2O3_J_mol * DRI_Al2O3_n_kmol
        )  # kJ Al2O3 BF pellets, '12. EAF Mass & Energy Balance!G268'

        scrap_Fe_J_mol = 0.0  # H (J/mol) Fe, '12. EAF Mass & Energy Balance!D269' > '14. Enthalpy Calculations!C113'
        scrap_Fe_kg = (
            mass_scrap_per_tLS * scrap_composition["Fe"]
        )  # kg Fe, '12. EAF Mass & Energy Balance!F269'
        scrap_Fe_n_kmol = scrap_Fe_kg / 55.80  # kmol Fe, '12. EAF Mass & Energy Balance!E269'
        scrap_Fe_kJ = (
            scrap_Fe_J_mol * scrap_Fe_n_kmol
        )  # kJ Fe, '12. EAF Mass & Energy Balance!G269'

        scrap_SiO2_J_mol = (
            -9.0830e05
        )  # H (J/mol) SiO2, '12. EAF Mass & Energy Balance!D270' > '14. Enthalpy Calculations!C207'
        scrap_SiO2_kg = mass_SiO2_scrap_per_tLS  # kg SiO2, '12. EAF Mass & Energy Balance!F270' > '12. EAF Mass & Energy Balance!D201'
        scrap_SiO2_n_kmol = scrap_SiO2_kg / 60.0  # kmol SiO2, '12. EAF Mass & Energy Balance!E270'
        scrap_SiO2_kJ = (
            scrap_SiO2_J_mol * scrap_SiO2_n_kmol
        )  # kJ SiO2, '12. EAF Mass & Energy Balance!G270'

        flux_CaO_J_mol = (
            -6.3490e05
        )  # H (J/mol) CaO, '12. EAF Mass & Energy Balance!D271' > '14. Enthalpy Calculations!C93'
        flux_CaO_kg = mass_CaO_slag_per_tLS  # kg CaO, '12. EAF Mass & Energy Balance!F271' > '12. EAF Mass & Energy Balance!D207'
        flux_CaO_n_kmol = flux_CaO_kg / 56.0  # kmol CaO, '12. EAF Mass & Energy Balance!E271'
        flux_CaO_kJ = (
            flux_CaO_J_mol * flux_CaO_n_kmol
        )  # kJ CaO, '12. EAF Mass & Energy Balance!G271'

        flux_MgO_J_mol = (
            -6.0160e05
        )  # H (J/mol) MgO, '12. EAF Mass & Energy Balance!D272' > '14. Enthalpy Calculations!C181'
        flux_MgO_kg = mass_MgO_slag_per_tLS  # kg MgO, '12. EAF Mass & Energy Balance!F272' > '12. EAF Mass & Energy Balance!D212'
        flux_MgO_n_kmol = flux_MgO_kg / 40.0  # kmol MgO, '12. EAF Mass & Energy Balance!E272'
        flux_MgO_kJ = (
            flux_MgO_J_mol * flux_MgO_n_kmol
        )  # kJ MgO, '12. EAF Mass & Energy Balance!G272'

        O2_J_mol = 0.0  # H (J/mol) O2, '12. EAF Mass & Energy Balance!D273' > '14. Enthalpy Calculations!C220'
        O2_n_kmol = moles_O2_per_tLS  # kmol O2, '12. EAF Mass & Energy Balance!E273' > '12. EAF Mass & Energy Balance!D241'
        O2_n_kmol * 32.0  # kg O2, '12. EAF Mass & Energy Balance!F273'
        O2_kJ = O2_J_mol * O2_n_kmol  # kJ O2, '12. EAF Mass & Energy Balance!G273'

        C_J_mol = 0.0  # H (J/mol) C from NG and injected Carbon, '12. EAF Mass & Energy Balance!D274' > '14. Enthalpy Calculations!C69'
        C_kg = (
            mass_C_ng_per_tLS + mass_injected_carbon_per_tLS
        )  # kg C from NG and injected Carbon, '12. EAF Mass & Energy Balance!F274'
        C_n_kmol = (
            C_kg / 12.0
        )  # kmol C from NG and injected Carbon, '12. EAF Mass & Energy Balance!E274' > '12. EAF Mass & Energy Balance!D241'
        C_kJ = (
            C_J_mol * C_n_kmol
        )  # kJ C from NG and injected Carbon, '12. EAF Mass & Energy Balance!G274'

        total_EAF_DRI_inputs_kJ = (
            DRI_Fe_kJ
            + DRI_FeO_kJ
            + DRI_C_kJ
            + DRI_SiO2_kJ
            + DRI_Al2O3_kJ
            + scrap_Fe_kJ
            + scrap_SiO2_kJ
            + flux_CaO_kJ
            + flux_MgO_kJ
            + O2_kJ
            + C_kJ
        )

        # EAF Products
        # NOTE: Possibly replace these mole values with actual enthalpy calculations from excel sheet?
        # Steel, Slag, Off-gas
        steel_Fe_J_mol = 7.583849597377010e04  # H (J/mol) Fe in Steel product, '12. EAF Mass & Energy Balance!D279' > '14. Enthalpy Calculations!C364'
        steel_Fe_kg = mass_iron_per_tLS  # kg Fe in Steel product, '12. EAF Mass & Energy Balance!F279' > '12. EAF Mass & Energy Balance!D7'
        steel_Fe_kmol = (
            steel_Fe_kg / 55.80
        )  # kmol Fe in Steel product, '12. EAF Mass & Energy Balance!E279'
        steel_Fe_kJ = (
            steel_Fe_J_mol * steel_Fe_kmol
        )  # kJ Fe in Steel product, '12. EAF Mass & Energy Balance!G279'

        steel_C_J_mol = 3.2201451507069200e04  # H (J/mol) C in Steel product, '12. EAF Mass & Energy Balance!D280' > '14. Enthalpy Calculations!C371'
        steel_C_kg = mass_carbon_per_tLS  # kg C in Steel product, '12. EAF Mass & Energy Balance!F280' > '12. EAF Mass & Energy Balance!D8'
        steel_C_kmol = (
            steel_C_kg / 12.0
        )  # kmol C in Steel product, '12. EAF Mass & Energy Balance!E280'
        steel_C_kJ = (
            steel_C_J_mol * steel_C_kmol
        )  # kJ C in Steel product, '12. EAF Mass & Energy Balance!G280'

        slag_FeO_kg = mass_FeO_slag_per_tLS  # kg FeO in slag product, '12. EAF Mass & Energy Balance!F281' > '12. EAF Mass & Energy Balance!D213'
        slag_SiO2_kg = mass_SiO2_slag_per_tLS  # kg SiO2 in slag product, '12. EAF Mass & Energy Balance!F282' > '12. EAF Mass & Energy Balance!D205'
        slag_Al2O3_kg = mass_Al2O3_slag_per_tLS  # kg Al2O3 in slag product, '12. EAF Mass & Energy Balance!F283' > '12. EAF Mass & Energy Balance!D206'
        slag_CaO_kg = mass_CaO_slag_per_tLS  # kg CaO in slag product, '12. EAF Mass & Energy Balance!F284' > '12. EAF Mass & Energy Balance!D207'
        slag_MgO_kg = mass_MgO_slag_per_tLS  # kg MgO in slag product, '12. EAF Mass & Energy Balance!F285' > '12. EAF Mass & Energy Balance!D212'
        BF_pellets_MJ_kg = -8.377283654515320e00  # H (MJ/kg) BF grade pellets estimated enthalpy of liquid slag (Bjorkvall approach), '14. Enthalpy Calculations!J14'
        DR_pellets_MJ_kg = -8.382019633585080e00  # H (MJ/kg) BF grade pellets estimated enthalpy of liquid slag (Bjorkvall approach), '14. Enthalpy Calculations!J14'

        if pellet_grade == "BF":
            slag_total_kJ = (
                BF_pellets_MJ_kg
                * (slag_FeO_kg + slag_SiO2_kg + slag_Al2O3_kg + slag_CaO_kg + slag_MgO_kg)
            ) * 1000
        if pellet_grade == "DR":
            slag_total_kJ = (
                DR_pellets_MJ_kg
                * (slag_FeO_kg + slag_SiO2_kg + slag_Al2O3_kg + slag_CaO_kg + slag_MgO_kg)
            ) * 1000

        off_gas_CO_J_mol = -5.887443663594190e04  # H (J/mol) CO in off-gas product, '12. EAF Mass & Energy Balance!D286' > '14. Enthalpy Calculations!C322'
        off_gas_CO_kmol = moles_CO_injected_per_tLS  # kmol CO in off-gas product, '12. EAF Mass & Energy Balance!E286' > '12. EAF Mass & Energy Balance!D239'
        (
            mass_CO_injected_per_tLS + mass_CO_DRI_per_tLS
        )  # kg CO in off-gas product, '12. EAF Mass & Energy Balance!F286'
        off_gas_CO_kJ = (
            off_gas_CO_J_mol * off_gas_CO_kmol
        )  # kJ CO in off-gas product, '12. EAF Mass & Energy Balance!G286'

        off_gas_CO2_J_mol = -3.10933969795530e05  # H (J/mol) CO2 in off-gas product, '12. EAF Mass & Energy Balance!D287' > '14. Enthalpy Calculations!C327'
        off_gas_CO2_kmol = moles_CO2_ng_per_tLS  # kmol CO2 in off-gas product, '12. EAF Mass & Energy Balance!E287' > '12. EAF Mass & Energy Balance!D231'
        off_gas_CO2_kJ = (
            off_gas_CO2_J_mol * off_gas_CO2_kmol
        )  # kJ CO2 in off-gas product, '12. EAF Mass & Energy Balance!G287'

        total_EAF_DRI_products_kJ = (
            steel_Fe_kJ + steel_C_kJ + slag_total_kJ + off_gas_CO_kJ + off_gas_CO2_kJ
        )

        EAF_DRI_energy_consumption_kJ_tHM = (
            total_EAF_DRI_products_kJ - total_EAF_DRI_inputs_kJ
        )  # (kJ/tHM) Energy Consumption of EAF, '12. EAF Mass & Energy Balance!G290'
        EAF_DRI_energy_consumption_kWh_tHM = (
            EAF_DRI_energy_consumption_kJ_tHM / kWh_to_kJ
        )  # (kWh/tHM) Energy Consumption of EAF, '12. EAF Mass & Energy Balance!G291'

        # NOTE EAF_scarp_heat_loss_adjustment_abs comes from EAF scrap only performance model
        # NOTE: Need to update below to possibly call EAF scrap only model here and calculate, or pre-calculate in other workflows
        EAF_DRI_heat_loss_adjustment_abs = EAF_scrap_heat_loss_adjustment_abs  # (kWh/tHM) EAF DRI absolute heat loss adjustment, '12. EAF Mass & Energy Balance!293' > '12. EAF Mass & Energy Balance!G148'
        (
            EAF_DRI_heat_loss_adjustment_abs / 470.00
        )  # % EAF DRI heat loss adjustment, '12. EAF Mass & Energy Balance!G294'

        EAF_DRI_energy_consumption_w_heat_loss_kWh_tHM = (
            EAF_DRI_energy_consumption_kWh_tHM + EAF_DRI_heat_loss_adjustment_abs
        )  # (kWh/tHM) Total EAF with scrap energy consumption with heat loss adjustment, '12. EAF Mass & Energy Balance!G295'

        # 5. Electric Arc Furnace > Including DRI in Feed
        # Electric Arc Furnace (EAF) with DRI process Inputs (Feedstocks)
        if pellet_grade == "BF":
            oxygen = nm3_O2_per_tLS  # Nm^3 O2/ton Steel, '12. EAF Mass & Energy Balance!D242' > '5. Electric Arc Furnace!C30'
            electricity = EAF_DRI_energy_consumption_w_heat_loss_kWh_tHM  # kWh/ton Hot Metal / Liquid Steel?, '12. EAF Mass & Energy Balance!G296' >'5. Electric Arc Furnace!C31'
            scrap_steel = (
                mass_scrap_per_tLS / 1000
            )  # ton, '5. Electric Arc Furnace!C34' > '12. EAF Mass & Energy Balance!D196/1000'
            coal = (
                mass_injected_carbon_per_tLS / 0.806 / 1000
            )  # ton, assum 0.806 tonC/tonCoal, '5. Electric Arc Furnace!C35' > '12. EAF Mass & Energy Balance!D228/0.806/1000'
            burnt_doloma = (
                mass_doloma / 1000
            )  # ton, '5. Electric Arc Furnace!C36' > '12. EAF Mass & Energy Balance!D257/1000'
            burnt_lime = (
                mass_lime / 1000
            )  # ton, '5. Electric Arc Furnace!C37' > '12. EAF Mass & Energy Balance!D258/1000'
        # TODO: update with calculations from mass balance for DR pellets
        if pellet_grade == "DR":
            oxygen = "12. EAF Mass & Energy Balance!I273"  # Nm^3/ton Steel, '5. Electric Arc Furnace!C39'
            electricity = "12. EAF Mass & Energy Balance!J296"  # kWh, '5. Electric Arc Furnace!C40'
            scrap_steel = "12. EAF Mass & Energy Balance!J296"  # ton, '5. Electric Arc Furnace!C43'
            coal = "12. EAF Mass & Energy Balance!E228/0.806/1000"  # ton, assum 0.806 tonC/tonCoal, '5. Electric Arc Furnace!C44'
            burnt_doloma = (
                "12. EAF Mass & Energy Balance!E257/1000"  # ton, '5. Electric Arc Furnace!C45'
            )
            burnt_lime = (
                "12. EAF Mass & Energy Balance!E258/1000"  # ton, '5. Electric Arc Furnace!C46'
            )

        # set outputs
        # Input feedstocks for EAF Fed with DRI Directly (No ESF)
        outputs["oxygen"] = oxygen  # Total Nm^3 O2 per ton steel
        outputs["electricity"] = electricity  # Total kWh electricity per ton steel
        outputs["natural_gas"] = natural_gas  # Total MMBtu NG per ton steel
        outputs["electrodes"] = electrodes  # Total kg Electrodes per ton steel
        outputs["scrap steel"] = scrap_steel  # Total ton scrap per ton steel
        outputs["coal"] = coal  # Total ton coal per ton steel
        outputs["burnt doloma"] = burnt_doloma  # Total ton burnt doloma per ton steel
        outputs["burnt_lime"] = burnt_lime  # Total ton burnt lime per ton steel

        # Output for EAF Fed with DRI Directly (No ESF)
        outputs["mass slag"] = mass_slag_per_tDRI  # Total kg slag per ton DRI
        outputs["mass MgO slag"] = mass_MgO_slag_per_tDRI  # Total kg MgO in slag per ton DRI
        outputs["mass FeO slag"] = mass_FeO_slag_per_tDRI  # Total kg FeO in slag per ton DRI
        outputs["mass FeO DRI"] = mass_FeO_DRI_per_tDRI  # Total kg FeO from DRI per ton DRI
        outputs["added mass FeO for slag"] = (
            add_mass_FeO_needed  # Total kg FeO from DRI per ton DRI
        )
        outputs["mass Fe to FeO"] = (
            mass_Fe_to_FeO  # Total kg Fe consumed to produce FeO per ton DRI
        )
        outputs["mass Fe from DRI"] = mass_Fe_DRI_per_tDRI  # Total kg Fe from DRI per ton DRI
        outputs["mass Fe from scrap"] = mass_Fe_scrap_per_tDRI  # Total kg Fe from scrap per ton DRI
        outputs["mass steel"] = (
            mass_steel_per_tDRI  # Total kg Steel formed from DRI + scrap per tDRI
        )