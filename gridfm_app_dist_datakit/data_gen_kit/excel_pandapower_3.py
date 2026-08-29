import pandas as pd
import pandapower as pp
import numpy as np
import re
from typing import Dict, List, Tuple


class ExcelToPandapowerImporter2:
    """
    Import power system data from Excel format to pandapower network
    """

    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path
        self.net = None
        self.bus_name_to_index = {}  # Mapping from bus names to pandapower indices
        self.linecode_data = {}  # Store line code parameters
        self.base_data = {}  # Store base values from Ref sheet

    def load_excel_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all sheets from Excel file
        """
        try:
            data = {}
            data['ref'] = pd.read_excel(self.excel_file_path, sheet_name='Ref', header=None)
            data['nodes'] = pd.read_excel(self.excel_file_path, sheet_name='Nodes')
            data['lines'] = pd.read_excel(self.excel_file_path, sheet_name='Lines')
            data['trafos'] = pd.read_excel(self.excel_file_path, sheet_name='Trafos')
            data['loads'] = pd.read_excel(self.excel_file_path, sheet_name='Loads')
            data['linecodes'] = pd.read_excel(self.excel_file_path, sheet_name='LineCodes')

            print("Successfully loaded Excel data:")
            print(f"  - Ref: {len(data['ref'])} parameters")
            print(f"  - Nodes: {len(data['nodes'])} buses")
            print(f"  - Lines: {len(data['lines'])} records")
            print(f"  - Transformers: {len(data['trafos'])} records")
            print(f"  - Loads: {len(data['loads'])} records")
            print(f"  - LineCodes: {len(data['linecodes'])} records")

            return data

        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None

    def parse_base_data(self, ref_df: pd.DataFrame):
        """
        Parse base data from Ref sheet
        """
        self.base_data = {}
        for idx, row in ref_df.iterrows():
            if pd.notna(row[0]) and pd.notna(row[1]):
                self.base_data[str(row[0]).strip()] = row[1]

        print("Base data parsed:")
        for key, value in self.base_data.items():
            print(f"  - {key}: {value}")

    def create_pandapower_network(self):
        """
        Create pandapower network from Excel data
        """
        # Load data from Excel
        data = self.load_excel_data()
        if data is None:
            return None

        # Parse base data
        self.parse_base_data(data['ref'])

        # Get base values
        base_mva = float(self.base_data.get('BaseMVA', 100))
        v_pri = float(self.base_data.get('V_pri', 11))
        v_sec = float(self.base_data.get('V_sec', 0.416))

        # Create empty network
        self.net = pp.create_empty_network(sn_mva=base_mva)

        # Load linecode data
        self._load_linecode_data(data['linecodes'])

        # Create buses from Nodes sheet
        self._create_buses_from_nodes(data['nodes'], v_pri, v_sec)

        # Create lines
        self._create_lines(data['lines'])

        # Create transformers
        self._create_transformers(data['trafos'])

        # Create loads
        self._create_loads(data['loads'])

        # Add external grid (assuming bus 0 as slack)
        self._add_external_grid()

        print(f"\nSuccessfully created pandapower network:")
        print(f"  - {len(self.net.bus)} buses")
        print(f"  - {len(self.net.line)} lines")
        print(f"  - {len(self.net.trafo)} transformers")
        print(f"  - {len(self.net.load)} loads")
        print(f"  - {len(self.net.ext_grid)} external grids")

        return self.net

    def _load_linecode_data(self, linecodes_df: pd.DataFrame):
        """
        Load line code data into dictionary for easy lookup
        """
        for idx, linecode in linecodes_df.iterrows():
            name = str(linecode['Name'])
            self.linecode_data[name] = {
                'r1': float(linecode['R1']),  # Positive sequence resistance (ohm/km)
                'x1': float(linecode['X1']),  # Positive sequence reactance (ohm/km)
                'r0': float(linecode.get('R0', linecode['R1'])),  # Zero sequence resistance
                'x0': float(linecode.get('X0', linecode['X1'])),  # Zero sequence reactance
                'c1': float(linecode.get('C1', 0)),  # Positive sequence capacitance (nF/km)
                'c0': float(linecode.get('C0', 0)),  # Zero sequence capacitance (nF/km)
                'units': str(linecode.get('Units', 'km'))
            }

        print(f"Loaded {len(self.linecode_data)} line codes")

    def _create_buses_from_nodes(self, nodes_df: pd.DataFrame, v_pri: float, v_sec: float):
        """
        Create pandapower buses from Nodes sheet
        """
        voltage_mapping = {
            'V_pri': v_pri,
            'V_sec': v_sec
        }

        for idx, node in nodes_df.iterrows():
            busname = str(node['Busname'])
            voltage_type = str(node['Voltage'])

            # Get voltage level from mapping
            vn_kv = voltage_mapping.get(voltage_type, v_sec)  # Default to secondary

            bus_idx = pp.create_bus(
                self.net,
                vn_kv=vn_kv,
                name=busname,
                max_vm_pu=1.05,
                min_vm_pu=0.95
            )
            self.bus_name_to_index[busname] = bus_idx

            # Store original bus number for reference
            self.net.bus.at[bus_idx, 'original_name'] = busname

    def _get_line_parameters(self, linecode: str, length: float, units: str) -> Tuple[float, float, float, float]:
        """
        Get line parameters from linecode data
        """
        if linecode not in self.linecode_data:
            print(f"Warning: Line code '{linecode}' not found. Using default values.")
            # Default values for missing line codes
            return 0.1, 0.1, 0, 0.4  # r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka

        params = self.linecode_data[linecode]

        # Convert length to km if needed
        if units.lower() == 'm':
            length_km = length / 1000  # meters to km
        elif units.lower() == 'ft':
            length_km = length * 0.0003048  # feet to km
        elif units.lower() == 'mi':
            length_km = length * 1.60934  # miles to km
        else:  # assume km
            length_km = length

        # Use positive sequence parameters for pandapower
        r_ohm_per_km = params['r1']
        x_ohm_per_km = params['x1']
        c_nf_per_km = params['c1']

        # Estimate max current based on typical values (this is approximate)
        # You might want to customize this based on your system
        if r_ohm_per_km > 0:
            max_i_ka = min(1.0, 10.0 / (r_ohm_per_km * 10))  # Rough estimation
        else:
            max_i_ka = 0.4

        return r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka, length_km

    def _create_lines(self, lines_df: pd.DataFrame):
        """
        Create pandapower lines from Lines sheet
        """
        for idx, line in lines_df.iterrows():
            try:
                bus1_name = str(line['Bus1'])
                bus2_name = str(line['Bus2'])

                if bus1_name not in self.bus_name_to_index or bus2_name not in self.bus_name_to_index:
                    print(f"Warning: Skipping line {line['Name']} - bus not found: {bus1_name} or {bus2_name}")
                    continue

                from_bus = self.bus_name_to_index[bus1_name]
                to_bus = self.bus_name_to_index[bus2_name]

                # Get line parameters
                linecode = str(line['LineCode']) if pd.notna(line['LineCode']) else '4c_70'
                length = float(line['Length']) if pd.notna(line['Length']) else 1.0
                units = str(line['Units']) if pd.notna(line['Units']) else 'm'

                r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka, length_km = self._get_line_parameters(
                    linecode, length, units)

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
                print(f"Created line {line['Name']}: {from_bus} to {to_bus}")
            except Exception as e:
                print(f"Error creating line {line['Name']}: {e}")

    def _create_transformers(self, trafos_df: pd.DataFrame):
        """
        Create pandapower transformers from Trafos sheet
        """
        for idx, trafo in trafos_df.iterrows():
            try:
                bus1_name = trafo['bus1']
                bus2_name = trafo['bus2']

                hv_bus = bus1_name
                lv_bus = bus2_name

                # Transformer parameters
                vn_hv_kv = float(trafo['kV_pri'])
                vn_lv_kv = float(trafo['kV_sec'])
                sn_mva = float(trafo['MVA'])

                # Impedance parameters
                vk_percent = float(trafo['%XHL'])
                vkr_percent = float(trafo['% resistance'])

                # Calculate shift angle based on winding connection
                shift_degree = self._get_shift_angle(
                    str(trafo.get('Conn_pri', 'Delta')),
                    str(trafo.get('Conn_sec', 'Wye'))
                )

                # Create transformer with all required parameters
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
                    shift_degree=shift_degree,
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

    def _get_shift_angle(self, conn_pri: str, conn_sec: str) -> float:
        """
        Calculate phase shift angle based on winding connections
        """
        conn_pri = conn_pri.lower()
        conn_sec = conn_sec.lower()

        # Common phase shifts for different connections
        if conn_pri == 'delta' and conn_sec == 'wye':
            return 30.0  # Delta-Wye connection typically has 30 degree shift
        elif conn_pri == 'wye' and conn_sec == 'delta':
            return -30.0  # Wye-Delta connection
        elif conn_pri == 'delta' and conn_sec == 'delta':
            return 0.0  # Delta-Delta
        elif conn_pri == 'wye' and conn_sec == 'wye':
            return 0.0  # Wye-Wye
        else:
            return 0.0  # Default no shift

    def _create_loads(self, loads_df: pd.DataFrame):
        """
        Create pandapower loads from Loads sheet
        """
        # Group loads by bus
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
                if pf > 0:
                    kva = kw / pf
                    kvar = np.sqrt(kva ** 2 - kw ** 2) if kva > kw else 0
                else:
                    kvar = 0

                # Convert to MW and MVAR
                p_mw = kw / 1000
                q_mvar = kvar / 1000

                # Aggregate loads at same bus
                if bus_idx in load_groups:
                    load_groups[bus_idx] = (
                        load_groups[bus_idx][0] + p_mw,
                        load_groups[bus_idx][1] + q_mvar
                    )
                else:
                    load_groups[bus_idx] = (p_mw, q_mvar)

            except Exception as e:
                print(f"Error processing load {load['Name']}: {e}")

        # Create pandapower loads
        for bus_idx, (p_mw, q_mvar) in load_groups.items():
            try:
                bus_name = self.net.bus.at[bus_idx, 'name']
                pp.create_load(
                    self.net,
                    bus=bus_idx,
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    name=f"Load_{bus_name}",
                    scaling=1.0
                )
                print(f"Created load at bus {bus_name}: {p_mw:.4f} MW, {q_mvar:.4f} MVAR")
            except Exception as e:
                print(f"Error creating load at bus {bus_idx}: {e}")

    def _add_external_grid(self):
        """
        Add external grid (slack bus) at bus 0
        """
        if '0' in self.bus_name_to_index:
            slack_bus_idx = self.bus_name_to_index['0']
            pp.create_ext_grid(
                self.net,
                bus=slack_bus_idx,
                vm_pu=1.0,
                va_degree=0.0,
                name="External_Grid"
            )
            print(f"Added external grid at bus: 0")
        else:
            # Find first bus as fallback
            if len(self.net.bus) > 0:
                slack_bus_idx = self.net.bus.index[0]
                pp.create_ext_grid(
                    self.net,
                    bus=slack_bus_idx,
                    vm_pu=1.0,
                    va_degree=0.0,
                    name="External_Grid"
                )
                print(f"Added external grid at bus: {self.net.bus.at[slack_bus_idx, 'name']}")
            else:
                print("Warning: No buses found for external grid")

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

            # Display some results
            print("\nBus results (first 10):")
            print(net.res_bus[['vm_pu', 'va_degree', 'p_mw', 'q_mvar']].head(10))

            print("\nLine loading (first 10):")
            print(net.res_line[['p_from_mw', 'q_from_mvar', 'loading_percent']].head(10))

            return True
        except Exception as e:
            print(f"Power flow failed: {e}")
            return False

"""
# Usage example
if __name__ == "__main__":
    # Initialize importer
    importer = ExcelToPandapowerImporter2("European_LV.xlsx")

    # Create pandapower network
    net = importer.create_pandapower_network()

    if net is not None:
        # Get bus mapping
        bus_mapping = importer.get_bus_mapping()
        print("\nBus mapping:")
        print(bus_mapping.head(10))

        # Run power flow
        importer.run_power_flow()
"""