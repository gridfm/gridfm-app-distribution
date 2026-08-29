import pandas as pd
import pandapower as pp
import numpy as np
import re
from typing import Dict, List, Tuple


class ExcelToPandapowerImporter:
    """
    Import power system data from Excel format to pandapower network
    """

    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path
        self.net = None
        self.bus_name_to_index = {}  # Mapping from bus names to pandapower indices
        self.linecode_data = {}  # Store line code parameters

    def load_excel_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all sheets from Excel file
        """
        try:
            data = {}
            data['lines'] = pd.read_excel(self.excel_file_path, sheet_name='Lines')
            data['trafos'] = pd.read_excel(self.excel_file_path, sheet_name='Trafos')
            data['loads'] = pd.read_excel(self.excel_file_path, sheet_name='Loads')
            data['ref'] = pd.read_excel(self.excel_file_path, sheet_name='Ref')

            print("Successfully loaded Excel data:")
            print(f"  - Lines: {len(data['lines'])} records")
            print(f"  - Transformers: {len(data['trafos'])} records")
            print(f"  - Loads: {len(data['loads'])} records")

            return data

        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None

    def extract_all_buses(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Extract all unique bus names from lines and transformers
        """
        buses = set()
        # From lines
        if 'lines' in data:
            buses.update(data['lines']['Bus1'].dropna().unique())
            buses.update(data['lines']['Bus2'].dropna().unique())

        # From transformers
        if 'trafos' in data:
            buses.update(data['trafos']['bus1'].dropna().unique())
            buses.update(data['trafos']['bus2'].dropna().unique())

        # From loads
        if 'loads' in data:
            buses.update(data['loads']['Bus'].dropna().unique())

        bus_list = sorted(list(buses), key=self._natural_sort_key)
        print(f"Found {len(bus_list)} unique buses")
        return bus_list

    def _natural_sort_key(self, s):
        """
        Natural sort key for bus names (P1, P2, P10 instead of P1, P10, P2)
        """
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', str(s))]

    def create_pandapower_network(self):
        """
        Create pandapower network from Excel data
        """
        # Load data from Excel
        data = self.load_excel_data()
        if data is None:
            return None

        # Create empty network
        ref_df = data['ref']
        base_mva = ref_df.loc[ref_df.iloc[:, 0] == 'BaseMVA', ref_df.columns[1]].values[0]

        self.net = pp.create_empty_network(sn_mva=base_mva)

        # Extract all buses and create them
        all_buses = self.extract_all_buses(data)
        self._create_buses(all_buses)

        # Create lines
        self._create_lines(data['lines'])

        # Create transformers
        self._create_transformers(data['trafos'])

        # Create loads
        self._create_loads(data['loads'])

        # Add external grid (assuming first primary bus as slack)
        self._add_external_grid()

        print(f"\nSuccessfully created pandapower network:")
        print(f"  - {len(self.net.bus)} buses")
        print(f"  - {len(self.net.line)} lines")
        print(f"  - {len(self.net.trafo)} transformers")
        print(f"  - {len(self.net.load)} loads")
        print(f"  - {len(self.net.ext_grid)} external grids")

        return self.net

    def _create_buses(self, bus_names: List[str]):
        """
        Create pandapower buses
        """
        # Load data from Excel
        data = self.load_excel_data()
        if data is None:
            return None

        # Open ref sheet
        ref_df = data['ref']

        #Set bus voltages
        base_voltage_kv = ref_df.loc[ref_df.iloc[:, 0] == 'BasekV', ref_df.columns[1]].values[0]
        primary_voltage_kv = ref_df.loc[ref_df.iloc[:, 0] == 'Vpri', ref_df.columns[1]].values[0]
        secondary_voltage_kv = ref_df.loc[ref_df.iloc[:, 0] == 'Vsec', ref_df.columns[1]].values[0]
        tertiary_voltage_kv = ref_df.loc[ref_df.iloc[:, 0] == 'Vsec2', ref_df.columns[1]].values[0]
        base_voltage_buses = ["P1", "P2", "P3", "P4", "P7", "P8"]

        for bus_name in bus_names:
            if bus_name in base_voltage_buses:
                vn_kv = base_voltage_kv
            elif str(bus_name).startswith('P'):
                vn_kv = primary_voltage_kv
            elif bus_name.startswith("S"):
                try:
                    number = int(bus_name[1:])  # after the "S"
                    if 145 <= number <= 192:
                        vn_kv = tertiary_voltage_kv
                    else:
                        vn_kv = secondary_voltage_kv
                except ValueError:
                    vn_kv = default_voltage_kv

            bus_idx = pp.create_bus(
                self.net,
                vn_kv=vn_kv,
                name=str(bus_name),
                max_vm_pu=1.05,
                min_vm_pu=0.95
            )
            self.bus_name_to_index[str(bus_name)] = bus_idx

    def _get_line_parameters(self, linecode: str, length: float, units: str) -> Tuple[float, float, float]:
        """
        Get line parameters
        """
        # Convert length to km
        if units == 'ft':
            length_km = length * 0.0003048  # feet to km
        elif units == 'mi':
            length_km = length * 1.60934  # miles to km
        else:  # set any other length to km
            length_km = length

        # Line parameters based on typical values
        linecode_params = {
            'Trans': {
                'r_ohm_per_km': 0.1,
                'x_ohm_per_km': 0.38,
                'c_nf_per_km': 9,
                'max_i_ka': 0.45
            },
            'Pri': {
                'r_ohm_per_km': 0.07,
                'x_ohm_per_km': 0.12,
                'c_nf_per_km': 300,
                'max_i_ka': 0.5
            },
            'Sec': {
                'r_ohm_per_km': 0.043,
                'x_ohm_per_km': 0.11,
                'c_nf_per_km': 320,
                'max_i_ka': 0.35
            },
            'Switch': {
                'r_ohm_per_km': 0.0,
                'x_ohm_per_km': 0.0,
                'c_nf_per_km': 500,
                'max_i_ka': 0.5
            },
        }

        if linecode in linecode_params:
            params = linecode_params[linecode]
            return (params['r_ohm_per_km'], params['x_ohm_per_km'],
                    params['c_nf_per_km'], params['max_i_ka'], length_km)

    def _create_lines(self, lines_df: pd.DataFrame):
        """
        Create pandapower lines from Lines sheet
        """
        for idx, line in lines_df.iterrows():
            try:
                bus1_name = str(line['Bus1'])
                bus2_name = str(line['Bus2'])

                if bus1_name not in self.bus_name_to_index or bus2_name not in self.bus_name_to_index:
                    print(f"Warning: Skipping line {line['Name']} - bus not found")
                    continue

                from_bus = self.bus_name_to_index[bus1_name]
                to_bus = self.bus_name_to_index[bus2_name]

                # Get line parameters
                linecode = str(line['LineCode']) if pd.notna(line['LineCode']) else 'Pri'
                length = float(line['Length']) if pd.notna(line['Length']) else 1.0
                units = str(line['Units']) if pd.notna(line['Units']) else 'ft'

                r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka, length_km = self._get_line_parameters(
                    linecode, length, units)

                # Determine if it's a switch (zero impedance)
                if 'sw' in str(line['Name']).lower() or linecode == 'Switch':
                    # Create switch
                    pp.create_switch(
                        self.net,
                        bus=from_bus,
                        element=to_bus,
                        et='b',  # bus-bus switch
                        type="CB",
                        closed=True,
                        name=str(line['Name'])
                    )
                elif linecode == 'Trans' or linecode == 'Pri' or linecode == 'Sec':
                    # Create line
                    pp.create_line_from_parameters(
                        self.net,
                        from_bus=from_bus,
                        to_bus=to_bus,
                        length_km=length_km,
                        r_ohm_per_km=r_ohm_per_km,
                        x_ohm_per_km=x_ohm_per_km,
                        c_nf_per_km=c_nf_per_km,
                        max_i_ka=max_i_ka,
                        name=str(line['Name']),
                        in_service=True
                    )
                    print(f"Created Line {line['Name']}:{from_bus} to {to_bus} ")
            except Exception as e:
                print(f"Error creating line {line['Name']}: {e}")

    def _create_transformers(self, trafos_df: pd.DataFrame):
        """
        Create pandapower transformers with all required parameters
        """
        for idx, trafo in trafos_df.iterrows():
            try:
                bus1_name = str(trafo['bus1'])
                bus2_name = str(trafo['bus2'])

                if bus1_name not in self.bus_name_to_index or bus2_name not in self.bus_name_to_index:
                    print(f"Warning: Skipping transformer {trafo['Name']} - bus not found")
                    continue

                hv_bus = self.bus_name_to_index[bus1_name]
                lv_bus = self.bus_name_to_index[bus2_name]

                # Transformer parameters
                vn_hv_kv = float(trafo['V_pri']) / 1000  # Convert V to kV
                vn_lv_kv = float(trafo['V_sec']) / 1000  # Convert V to kV
                sn_mva = float(trafo['kVA']) / 1000  # Convert kVA to MVA

                # Impedance parameters
                vk_percent = float(trafo.get('%XHL', 7.0))
                vkr_percent = float(trafo.get('%RHL', 0.7))

                # Use create_transformer_from_parameters with all required fields
                pp.create_transformer_from_parameters(
                    self.net,
                    hv_bus=hv_bus,
                    lv_bus=lv_bus,
                    sn_mva=sn_mva,
                    vn_hv_kv=vn_hv_kv,
                    vn_lv_kv=vn_lv_kv,
                    vkr_percent=vkr_percent,
                    vk_percent=vk_percent,
                    pfe_kw=0.0,  # Iron losses (kW)
                    i0_percent=0.0,  # No-load current (%)
                    shift_degree=0.0,  # Phase shift
                    tap_side='hv',  # Tap changer side
                    tap_neutral=0,  # Neutral tap position
                    tap_min=-2,  # Minimum tap position
                    tap_max=2,  # Maximum tap position
                    tap_step_percent=1.0,  # Tap step size in %
                    tap_pos=0,  # Current tap position
                    name=str(trafo['Name'])
                )

                print(f"Created transformer {trafo['Name']}: {sn_mva} MVA, {vn_hv_kv}/{vn_lv_kv} kV")

            except Exception as e:
                print(f"Error creating transformer {trafo['Name']}: {e}")

    def _create_loads(self, loads_df: pd.DataFrame):
        """
        Create pandapower loads from Loads sheet
        """
        # Group loads by bus and phase
        load_groups = {}

        for idx, load in loads_df.iterrows():
            try:
                bus_name = str(load['Bus'])

                if bus_name not in self.bus_name_to_index:
                    print(f"Warning: Skipping load {load['Name']} - bus {bus_name} not found")
                    continue

                bus_idx = self.bus_name_to_index[bus_name]

                # Get load parameters
                kw = float(load['kW']) if pd.notna(load['kW']) else 0.0
                pf = float(load['PF']) if pd.notna(load['PF']) else 0.9

                # Calculate kVAR
                kva = kw / pf if pf > 0 else kw
                kvar = np.sqrt(kva ** 2 - kw ** 2) if kva > kw else 0

                # Convert to MW and MVAR
                p_mw = kw / 1000
                q_mvar = kvar / 1000

                # Create load key for grouping
                load_key = (bus_idx, str(load.get('phases', 'ABC')))

                if load_key in load_groups:
                    # Aggregate with existing load
                    load_groups[load_key] = (
                        load_groups[load_key][0] + p_mw,
                        load_groups[load_key][1] + q_mvar
                    )
                else:
                    # Create new load entry
                    load_groups[load_key] = (p_mw, q_mvar)

            except Exception as e:
                print(f"Error processing load {load['Name']}: {e}")

        # Create pandapower loads
        for (bus_idx, phases), (p_mw, q_mvar) in load_groups.items():
            try:
                pp.create_load(
                    self.net,
                    bus=bus_idx,
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    name=f"Load_{self.net.bus.at[bus_idx, 'name']}_{phases}",
                    scaling=1.0
                )
            except Exception as e:
                print(f"Error creating load at bus {bus_idx}: {e}")

    def _add_external_grid(self):
        """
        Add external grid (slack bus)
        Assuming first primary bus as slack
        """
        # Find first primary bus
        slack_bus_idx = None
        for bus_idx, bus in self.net.bus.iterrows():
            if bus['name'].startswith('P'):
                slack_bus_idx = bus_idx
                break

        if slack_bus_idx is not None:
            pp.create_ext_grid(
                self.net,
                bus=slack_bus_idx,
                vm_pu=1.0,
                va_degree=0.0,
                name="External_Grid"
            )
            print(f"Added external grid at bus: {self.net.bus.at[slack_bus_idx, 'name']}")
        else:
            print("Warning: No suitable slack bus found for external grid")

    def get_bus_mapping(self) -> pd.DataFrame:
        """
        Get bus name to index mapping
        """
        mapping_data = []
        for bus_name, bus_idx in self.bus_name_to_index.items():
            mapping_data.append({
                'bus_name': bus_name,
                'pandapower_index': bus_idx,
                'voltage_kv': self.net.bus.at[bus_idx, 'vn_kv']
            })
        return pd.DataFrame(mapping_data)

    def run_power_flow(self):
        """
        Run power flow analysis
        """
        if self.net is None:
            print("Network not created yet")
            return False

        try:
            pp.runpp(self.net)
            print("Power flow completed successfully")
            return True
        except Exception as e:
            print(f"Power flow failed: {e}")
            return False

"""
# Usage example
if __name__ == "__main__":
    # Initialize importer
    importer = ExcelToPandapowerImporter("IEEE-342.xlsx")

    # Create pandapower network
    net = importer.create_pandapower_network()

    if net is not None:
        # Get bus mapping
        bus_mapping = importer.get_bus_mapping()
        print("\nBus mapping:")
        print(bus_mapping)

        # Run power flow
        importer.run_power_flow()

        # Display results
        print("\nBus results:")
        print(net.res_bus[['vm_pu', 'va_degree', 'p_mw', 'q_mvar']])

        print("\nLine results:")
        print(net.res_line[['p_from_mw', 'q_from_mvar', 'loading_percent']])
"""