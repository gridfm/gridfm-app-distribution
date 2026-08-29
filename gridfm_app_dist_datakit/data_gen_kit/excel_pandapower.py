import pandas as pd
import pandapower as pp
import numpy as np
import openpyxl

def load_excel_data_to_dicts(excel_file_path):
    """
    Load network data from Excel file into dictionaries
    
    Parameters:
    -----------
    excel_file_path : str
        Path to the Excel file
    
    Returns:
    --------
    dict
        Dictionary containing all network data as DataFrames
    """
    data_dict = {}
    
    try:
        # Load all sheets
        data_dict['nodes'] = pd.read_excel(excel_file_path, sheet_name='nodes')
        data_dict['branches'] = pd.read_excel(excel_file_path, sheet_name='branches')
        data_dict['gens'] = pd.read_excel(excel_file_path, sheet_name='gens')
        data_dict['ref'] = pd.read_excel(excel_file_path, sheet_name='Ref', header=None)
        
        print("Successfully loaded data from Excel file")
        return data_dict

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def create_pandapower_network(data_dict):
    """
    Create a pandapower network from the loaded data dictionaries
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing network data
        
    Returns:
    --------
    pandapower network
        Created pandapower network
    """
    
    # Extract base values from ref sheet
    ref_df = data_dict['ref']
    base_voltage = ref_df.loc[ref_df[0] == 'Voltage(kV)', 1].values[0]
    base_mva = ref_df.loc[ref_df[0] == 'BaseMVA', 1].values[0]
    
    # Create empty network
    net = pp.create_empty_network(sn_mva=base_mva)
    
    # Add buses
    for idx, node in data_dict['nodes'].iterrows():
        bus_id = pp.create_bus(
            net, 
            vn_kv=base_voltage,
            name=f"Bus_{int(node['bus_id'])}",
            min_vm_pu=node['Vmin p.u.'],
            max_vm_pu=node['Vmax p.u.']
        )
        # Store the original bus number mapping
        net.bus.at[bus_id, 'original_bus_no'] = int(node['bus_id'])
    
    # Create mapping from original bus numbers to pandapower bus indices
    bus_mapping = {}
    for idx, bus in net.bus.iterrows():
        bus_mapping[bus['original_bus_no']] = idx
    
    # Add external grid (reference bus)
    ref_nodes = data_dict['nodes'][data_dict['nodes']['type'] == 'reference']
    if not ref_nodes.empty:
        ref_bus_no = int(ref_nodes['bus_id'].iloc[0])
        ref_bus_idx = bus_mapping[ref_bus_no]
        pp.create_ext_grid(
            net, 
            bus=ref_bus_idx, 
            vm_pu=1.0,  # You might want to get this from your data
            va_degree=0.0
        )
    
    # Add loads
    load_nodes = data_dict['nodes'][data_dict['nodes']['type'] == 'PQ']
    for idx, node in load_nodes.iterrows():
        bus_idx = bus_mapping[int(node['bus_id'])]
        pp.create_load(
            net,
            bus=bus_idx,
            p_mw=node['Pd MW'],
            q_mvar=node['Qd MVAR'],
            name=f"Load_{int(node['bus_id'])}"
        )
    
    # Add lines (branches)
    for idx, branch in data_dict['branches'].iterrows():
        from_bus = bus_mapping[int(branch['from bus'])]
        to_bus = bus_mapping[int(branch['to bus'])]
        
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,  # Assuming length is 1km if not specified
            r_ohm_per_km=branch['r Ohm'],
            x_ohm_per_km=branch['x Ohm'],
            c_nf_per_km=0,  # Assuming no capacitance
            max_i_ka=1.0,   # Default value
            name=f"Line_{int(branch['branch no'])}",
            in_service=bool(branch['in service'])
        )
    
    # Add generators
    for idx, gen in data_dict['gens'].iterrows():
        bus_idx = bus_mapping[int(gen['bus_id'])]
        
        # Check if this is not the reference bus (external grid)
        if int(gen['bus_id']) != ref_bus_no:
            pp.create_gen(
                net,
                bus=bus_idx,
                p_mw=gen['Pmin MW'],  # Set to minimum active power
                min_p_mw=gen['Pmin MW'],
                max_p_mw=gen['Pmax MW'],
                min_q_mvar=gen['Qmin MVAR'],
                max_q_mvar=gen['Qmax MVAR'],
                vm_pu=1.0,  # Default voltage setpoint
                name=f"Gen_{int(gen['bus_id'])}",
                in_service=bool(gen['In service'])
            )
    
    print(f"Successfully created pandapower network with:")
    print(f"  - {len(net.bus)} buses")
    print(f"  - {len(net.load)} loads") 
    print(f"  - {len(net.line)} lines")
    print(f"  - {len(net.gen)} generators")
    print(f"  - {len(net.ext_grid)} external grid connections")
    
    return net

def run_load_flow_and_display_results(net):
    """
    Run load flow and display results
    
    Parameters:
    -----------
    net : pandapower network
        Network to run load flow on
    """
    try:
        # Run power flow
        pp.runpp(net)
        
        print("\nLoad Flow Results:")
        print("==================")
        
        # Display bus results
        print("\nBus Results:")
        print(net.res_bus[['vm_pu', 'va_degree', 'p_mw', 'q_mvar']])
        
        # Display line results
        print("\nLine Loading Results:")
        print(net.res_line[['p_from_mw', 'q_from_mvar', 'loading_percent']])
        
        return True
        
    except Exception as e:
        print(f"Error running load flow: {e}")
        return False

# Main execution function
def create_network_from_excel(excel_file_path, run_power_flow=True):
    """
    Main function to create pandapower network from Excel file
    
    Parameters:
    -----------
    excel_file_path : str
        Path to the Excel file
    run_power_flow : bool
        Whether to run power flow after creating network
        
    Returns:
    --------
    pandapower network
        Created network object
    """
    
    # Step 1: Load data from Excel to dictionaries
    print("Loading data from Excel...")
    data_dict = load_excel_data_to_dicts(excel_file_path)
    
    if data_dict is None:
        return None
    
    # Step 2: Create pandapower network
    print("Creating pandapower network...")
    net = create_pandapower_network(data_dict)
    
    # Step 3: Run load flow if requested
    if run_power_flow:
        print("Running load flow...")
        run_load_flow_and_display_results(net)
    
    return net

"""
# Usage:
if __name__ == "__main__":
    excel_file = "case33meshed.xlsx"
    
    # Create network
    network = create_network_from_excel(excel_file, run_power_flow=True)
    print(network.ext_grid)
    
    # Network summary
    if network is not None:
        print(f"\nNetwork created successfully!")
        print(f"Base MVA: {network.sn_mva}")
        print(f"Base voltage: {network.bus.vn_kv.iloc[0]} kV")
"""