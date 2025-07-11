import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt, savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat
import math


class DataInitializer:
    def __init__(self, cutoff=0.1, fs=2, order=2, window_size=30):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.window_size = window_size
        self.window_s = 25

    def low_pass_filter(self, data):
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def filter_dT_dV(self, x):
        if len(x) >= self.window_size:
            return savgol_filter(x, self.window_size, 5)
        else:
            return x
        
    def length_equalizer(self, input_list):    
        # Handle case where the entire input list is empty
        if not input_list:
            return np.array([])
            
        # Safely find the max length, ignoring any empty signals in the list
        valid_lengths = [len(signal) for signal in input_list if hasattr(signal, '__len__') and len(signal) > 0]
        if not valid_lengths:
            return np.array([[] for _ in input_list]) # Return list of empty arrays if all were empty
        max_length = max(valid_lengths)

        uniformed_signals = []
        for signal in input_list:
            # Check if the signal is empty or not a sequence
            if not hasattr(signal, '__len__') or len(signal) == 0:
                # If empty, append an array of zeros as a placeholder
                uniformed_signals.append(np.zeros(max_length))
                continue

            # Proceed with interpolation for valid signals
            original_indices = np.linspace(0, len(signal) - 1, num=len(signal))
            target_indices = np.linspace(0, len(signal) - 1, num=max_length)
            interpolated_signal = np.interp(target_indices, original_indices, signal)
            uniformed_signals.append(interpolated_signal)
            
        return np.array(uniformed_signals)
           
    def load_NASA(self, file_path, battery_name):
        """
        Loads and processes battery cycle data from the NASA .mat file format.

        This function extracts and separates time-series data for charge and discharge
        cycles, calculates a rich set of health indicators (HIs), and returns a
        standardized dictionary.

        Args:
            file_path (str): The path to the input .mat file (e.g., 'B0005.mat').
            battery_name (str): The name of the battery to access within the .mat file (e.g., 'B0005').

        Returns:
            dict: A dictionary containing the processed data with a standardized structure, including
                DataFrames, original and equalized time-series data, cycle lists, capacities, and HIs.
        """
        # --- Load data from .mat file ---
        mat = loadmat(file_path)

        # --- Initialize data storage ---
        ch_cap, dch_cap = [], []
        cycles = []

        # For building DataFrames
        all_charge_rows, all_discharge_rows = [], []

        # Unequalized time-series data
        time_ch, voltage_ch, temperature_ch, DTV_ch = [], [], [], []
        time_dch, voltage_dch, temperature_dch, DTV_dch = [], [], [], []

        # Health Indicator (HI) dictionaries
        HIs_ch = {
            'charge_time': [], 'temperature_std': [], 'voltage_std': [],
            'dv_dt_mean': [], 'dv_dt_std': [], 'mean_temp': [],
            'max_temp': [], 'voltage_integral': []
        }
        HIs_dch = {
            'discharge_time': [], 'temperature_std': [], 'voltage_std': [],
            'dv_dt_mean': [], 'dv_dt_std': [], 'mean_temp': [],
            'max_temp': [], 'voltage_integral': [], 'dv_dt_min':[]
        }

        # --- Loop through all cycle structures in the .mat file ---
        all_cycles_data = mat[battery_name][0, 0]['cycle'][0]
        charge_cycle_counter = 0
        discharge_cycle_counter = 0

        for i in range(len(all_cycles_data)):
            cycle_data = all_cycles_data[i]
            cycle_type = cycle_data['type'][0]

            # --- Process Charge Data ---
            if cycle_type == 'charge':
                charge_cycle_counter += 1
                data = cycle_data['data'][0, 0]

                # Extract time-series vectors 
                t_ch_vec = data['Time'][0]
                v_ch_vec = data['Voltage_measured'][0]
                i_ch_vec = data['Current_measured'][0]
                T_ch_vec = data['Temperature_measured'][0]

                # Store raw time-series data
                voltage_ch.append(v_ch_vec)
                temperature_ch.append(T_ch_vec)
                time_ch.append(t_ch_vec)

                # Calculate charge capacity by integrating current over time
                # (converts amp-seconds to amp-hours)
                charge_capacity = np.trapz(i_ch_vec, t_ch_vec) / 3600.0
                ch_cap.append(charge_capacity)

                # Calculate and store charge HIs
                if len(t_ch_vec) > 1:
                    HIs_ch['charge_time'].append(t_ch_vec[-1] - t_ch_vec[0])
                    HIs_ch['mean_temp'].append(np.mean(T_ch_vec))
                    HIs_ch['max_temp'].append(np.max(T_ch_vec))
                    HIs_ch['temperature_std'].append(np.std(T_ch_vec))
                    HIs_ch['voltage_std'].append(np.std(v_ch_vec))
                    HIs_ch['voltage_integral'].append(np.trapz(v_ch_vec, t_ch_vec))
                    dvdt_ch = np.diff(v_ch_vec) / np.diff(t_ch_vec)
                    HIs_ch['dv_dt_mean'].append(np.nanmean(dvdt_ch))
                    HIs_ch['dv_dt_std'].append(np.nanstd(dvdt_ch))
                    
                    # Calculate -dT/dV
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dt_dv_ch = np.diff(T_ch_vec) / np.diff(v_ch_vec)
                    dt_dv_ch[np.isinf(dt_dv_ch)] = np.nan
                    DTV_ch.append(-1 * pd.Series(dt_dv_ch).ffill().bfill().values)

                # Append rows for DataFrame
                for j in range(len(t_ch_vec)):
                    all_charge_rows.append({'Cycle': charge_cycle_counter, 'Step Type': 'CC-CV Chg', 'Time': t_ch_vec[j],
                                            'Voltage': v_ch_vec[j], 'Current': i_ch_vec[j], 'Temperature': T_ch_vec[j]})

            # --- Process Discharge Data ---
            elif cycle_type == 'discharge':
                discharge_cycle_counter += 1
                cycles.append(discharge_cycle_counter)
                data = cycle_data['data'][0, 0]

                # Extract time-series vectors and capacity 
                t_dch_vec = data['Time'][0]
                v_dch_vec = data['Voltage_measured'][0]
                i_dch_vec = data['Current_measured'][0]
                T_dch_vec = data['Temperature_measured'][0]
                discharge_capacity = data['Capacity'][0][0]
                dch_cap.append(discharge_capacity)

                # Store raw time-series data
                voltage_dch.append(v_dch_vec)
                temperature_dch.append(T_dch_vec)
                time_dch.append(t_dch_vec)

                # Calculate and store discharge HIs
                if len(t_dch_vec) > 1:
                    HIs_dch['discharge_time'].append(t_dch_vec[-1] - t_dch_vec[0])
                    HIs_dch['mean_temp'].append(np.mean(T_dch_vec))
                    HIs_dch['max_temp'].append(np.max(T_dch_vec))
                    HIs_dch['temperature_std'].append(np.std(T_dch_vec))
                    HIs_dch['voltage_std'].append(np.std(v_dch_vec))
                    HIs_dch['voltage_integral'].append(np.trapz(v_dch_vec, t_dch_vec))
                    dvdt_dch = np.diff(v_dch_vec) / np.diff(t_dch_vec)
                    HIs_dch['dv_dt_mean'].append(np.nanmean(dvdt_dch))
                    HIs_dch['dv_dt_std'].append(np.nanstd(dvdt_dch))
                    HIs_dch['dv_dt_min'].append(np.nanmin(dvdt_dch))
                    
                    # Calculate -dT/dV
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dt_dv_dch = np.diff(T_dch_vec) / np.diff(v_dch_vec)
                    dt_dv_dch[np.isinf(dt_dv_dch)] = np.nan
                    DTV_dch.append(-1 * pd.Series(dt_dv_dch).ffill().bfill().values)

                # Append rows for DataFrame
                for j in range(len(t_dch_vec)):
                    all_discharge_rows.append({'Cycle': discharge_cycle_counter, 'Step Type': 'CC DChg', 'Time': t_dch_vec[j],
                                            'Voltage': v_dch_vec[j], 'Current': i_dch_vec[j], 'Temperature': T_dch_vec[j], 'Capacity': discharge_capacity})

        # --- Create final data structures ---
        data_ch = pd.DataFrame(all_charge_rows)
        data_dch = pd.DataFrame(all_discharge_rows)

        data_original_ch = {'voltage': voltage_ch, 'temperature': temperature_ch, 'time': time_ch, 'DTV': DTV_ch}
        data_original_dch = {'voltage': voltage_dch, 'temperature': temperature_dch, 'time': time_dch, 'DTV': DTV_dch}

        data_eq_ch = {key: self.length_equalizer(val) for key, val in data_original_ch.items()}
        data_eq_dch = {key: self.length_equalizer(val) for key, val in data_original_dch.items()}
        
        print(f"Dataset successfully loaded from: {file_path}")
        # --- Final return dictionary matching the other methods ---
        return {
            'data_dch': data_dch,
            'data_ch': data_ch,
            'Original_dch': data_original_dch,
            'Original_ch': data_original_ch,
            'Equalized_dch': data_eq_dch,
            'Equalized_ch': data_eq_ch,
            'cycles': cycles,
            'dch_cap': dch_cap,
            'ch_cap': ch_cap,
            'HIs_dch': HIs_dch,
            'HIs_ch': HIs_ch
        }
    
    def load_oxford(self, file_path, cell_number):
        """
        Loads and processes battery cycle data from the Oxford .mat file format,
        matching the output structure of the `load_prepare` method.

        Args:
            file_path (str): The path to the input .mat file.
            cell_number (int): The cell number to extract from the file (e.g., 1 to 8).

        Returns:
            dict: A dictionary containing the processed data with a standardized structure.
        """
        # --- Load data from .mat file ---
        data = loadmat(file_path)

        cell_key = f'Cell{cell_number}'
        if cell_key not in data:
            raise ValueError(f"'{cell_key}' not found in the provided .mat file.")
        cell_data = data[cell_key]

        # --- Initialize data storage to match `load_prepare` structure ---
        ch_cap, dch_cap = [], []
        cycles = []
        
        # For building DataFrames
        all_charge_rows, all_discharge_rows = [], []

        # Unequalized time-series data
        time_ch, voltage_ch, temperature_ch, DTV_ch = [], [], [], []
        time_dch, voltage_dch, temperature_dch, DTV_dch = [], [], [], []

        # Health Indicator (HI) dictionaries
        HIs_ch = {
            'charge_time': [], 'temperature_std': [], 'voltage_std': [], 'dv_dt_mean': [],
            'dv_dt_std': [], 'mean_temp': [], 'max_temp': [], 'voltage_integral': [],
            'voltage_plateau_time': []
        }
        HIs_dch = {
            'discharge_time': [], 'temperature_std': [], 'voltage_std': [], 'dv_dt_mean': [],
            'dv_dt_std': [], 'mean_temp': [], 'max_temp': [], 'voltage_integral': [], 'dv_dt_min':[]
        }

        # --- Loop through all cycles in the cell data ---
        for cycle_key in sorted(cell_data.dtype.names):
            
            cycle_num = int(cycle_key[3:])
            cycles.append(cycle_num)

            cycle = cell_data[cycle_key][0, 0]
            
            # --- Process Charge Data (C1ch) ---
            if 'C1ch' in cycle.dtype.names and cycle['C1ch'].size > 0:
                C1ch_data = cycle['C1ch'][0, 0]
                T_ch_vec, q_ch_vec = C1ch_data['T'][0, 0].flatten(), C1ch_data['q'][0, 0].flatten()
                t_ch_vec, v_ch_vec = C1ch_data['t'][0, 0].flatten(), C1ch_data['v'][0, 0].flatten()

                # 1. Convert time from days to seconds
                t_ch_vec = t_ch_vec * 24 * 3600
                # 2. Reset time to start from zero for this cycle
                t_ch_vec = t_ch_vec - t_ch_vec[0]
                
                # Append data to lists
                temperature_ch.append(T_ch_vec)
                voltage_ch.append(v_ch_vec)
                time_ch.append(t_ch_vec)
                if len(q_ch_vec) > 0: ch_cap.append(q_ch_vec[-1])
                
                # Calculate and store charge HIs
                if len(t_ch_vec) > 1:
                    HIs_ch['charge_time'].append(t_ch_vec[-1] - t_ch_vec[0])
                    HIs_ch['mean_temp'].append(np.mean(T_ch_vec))
                    HIs_ch['max_temp'].append(np.max(T_ch_vec))
                    HIs_ch['temperature_std'].append(np.std(T_ch_vec))
                    HIs_ch['voltage_std'].append(np.std(v_ch_vec))
                    HIs_ch['voltage_integral'].append(np.trapz(v_ch_vec, t_ch_vec))
                    dvdt_ch = np.diff(v_ch_vec) / np.diff(t_ch_vec)
                    HIs_ch['dv_dt_mean'].append(np.nanmean(dvdt_ch))
                    HIs_ch['dv_dt_std'].append(np.nanstd(dvdt_ch))
                    plateau_voltage = 4.1
                    above_plateau_idx = np.where(v_ch_vec >= plateau_voltage)[0]
                    plateau_time = t_ch_vec[above_plateau_idx[0]] - t_ch_vec[0] if len(above_plateau_idx) > 0 else np.nan
                    HIs_ch['voltage_plateau_time'].append(plateau_time)
                    
                    # Calculate dT/dV
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # This line will still generate warnings, but the errstate context manages them
                        dt_dv_ch = np.diff(T_ch_vec) / np.diff(v_ch_vec)

                    # Replace any resulting infinity values with NaN (Not a Number)
                    dt_dv_ch[np.isinf(dt_dv_ch)] = np.nan

                    # Use pandas to easily forward-fill and then back-fill any NaN values
                    # This ensures there are no gaps in your data.
                    DTV_ch.append(-1 * pd.Series(dt_dv_ch).ffill().bfill().values)
                else:
                    print(f'Failed processing charge HIs for cycle {cycle_num}')
                
                # Append rows for DataFrame
                for i in range(len(t_ch_vec)):
                    all_charge_rows.append({'Cycle': cycle_num, 'Step Type': 'CC Chg', 'Time': t_ch_vec[i],
                                            'Voltage': v_ch_vec[i], 'Capacity': q_ch_vec[i], 'Temperature': T_ch_vec[i]})
            else:
                print(f'Processing charge data for cycle {cycle_num} failed')

            # --- Process Discharge Data (C1dc) ---
            if 'C1dc' in cycle.dtype.names and cycle['C1dc'].size > 0:
                C1dc_data = cycle['C1dc'][0, 0]
                T_dc_vec, q_dc_vec = C1dc_data['T'][0, 0].flatten(), C1dc_data['q'][0, 0].flatten()
                t_dc_vec, v_dc_vec = C1dc_data['t'][0, 0].flatten(), C1dc_data['v'][0, 0].flatten()

                # 1. Convert time from days to seconds
                t_dc_vec = t_dc_vec * 24 * 3600
                # 2. Reset time to start from zero for this cycle
                t_dc_vec = t_dc_vec - t_dc_vec[0]

                # Append data to lists
                temperature_dch.append(T_dc_vec)
                voltage_dch.append(v_dc_vec)
                time_dch.append(t_dc_vec)
                if len(q_dc_vec) > 0: dch_cap.append(q_dc_vec[-1])
                
                # Calculate and store discharge HIs
                if len(t_dc_vec) > 1:
                    HIs_dch['discharge_time'].append(t_dc_vec[-1] - t_dc_vec[0])
                    HIs_dch['mean_temp'].append(np.mean(T_dc_vec))
                    HIs_dch['max_temp'].append(np.max(T_dc_vec))
                    HIs_dch['temperature_std'].append(np.std(T_dc_vec))
                    HIs_dch['voltage_std'].append(np.std(v_dc_vec))
                    HIs_dch['voltage_integral'].append(np.trapz(v_dc_vec, t_dc_vec))
                    dvdt_dch = np.diff(v_dc_vec) / np.diff(t_dc_vec)
                    HIs_dch['dv_dt_mean'].append(np.nanmean(dvdt_dch))
                    HIs_dch['dv_dt_std'].append(np.nanstd(dvdt_dch))
                    HIs_dch['dv_dt_min'].append(np.nanmin(dvdt_dch))

                    
                    # Calculate dT/dV
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # This line will still generate warnings, but the errstate context manages them
                        dt_dv_dch = np.diff(T_dc_vec) / np.diff(v_dc_vec)

                    # Replace any resulting infinity values with NaN (Not a Number)
                    dt_dv_dch[np.isinf(dt_dv_dch)] = np.nan

                    # Use pandas to easily forward-fill and then back-fill any NaN values
                    # This ensures there are no gaps in your data.
                    DTV_dch.append(-1 * pd.Series(dt_dv_dch).ffill().bfill().values)
                else:
                    print(f'Failed processing discharge HIs for cycle {cycle_num}')

                # Append rows for DataFrame
                for i in range(len(t_dc_vec)):
                    all_discharge_rows.append({'Cycle': cycle_num, 'Step Type': 'CC DChg', 'Time': t_dc_vec[i],
                                            'Voltage': v_dc_vec[i], 'Capacity': q_dc_vec[i], 'Temperature': T_dc_vec[i]})
            else:
                print(f'Processing discharge data for cycle {cycle_num} failed')

        # --- Create final data structures ---
        data_ch = pd.DataFrame(all_charge_rows)
        data_dch = pd.DataFrame(all_discharge_rows)

        data_original_ch = {'voltage': voltage_ch, 'temperature': temperature_ch, 'time': time_ch, 'DTV': DTV_ch}
        data_original_dch = {'voltage': voltage_dch, 'temperature': temperature_dch, 'time': time_dch, 'DTV': DTV_dch}

        data_eq_ch = {key: self.length_equalizer(val) for key, val in data_original_ch.items()}
        data_eq_dch = {key: self.length_equalizer(val) for key, val in data_original_dch.items()}
        
        print(f"Cell #{cell_number} Data successfully loaded from: {file_path}")

        # --- Final return dictionary matching `load_prepare` ---
        return {
            'data_dch': data_dch,
            'data_ch': data_ch,
            'Original_dch': data_original_dch,
            'Original_ch': data_original_ch,
            'Equalized_dch': data_eq_dch,
            'Equalized_ch': data_eq_ch,
            'cycles': cycles,
            'dch_cap': dch_cap,
            'ch_cap': ch_cap,
            'HIs_dch': HIs_dch,
            'HIs_ch': HIs_ch
        }
    
    def load_Experimental(self, file_path):
        """
        Loads and preprocesses battery cycling data from a file.

        This method reads the raw data, resamples it, calculates differential
        temperature/voltage (dT/dV), separates charge and discharge cycles,
        and extracts numerous health indicators (HIs).

        Args:
            file_path (str): The path to the input CSV file.

        Returns:
            dict: A dictionary containing the processed data, including:
                  - 'data_dch', 'data_ch': DataFrames for discharge/charge steps.
                  - 'Original_dch', 'Original_ch': Dictionaries of unequalized time-series data.
                  - 'Equalized_dch', 'Equalized_ch': Dictionaries of equalized time-series data.
                  - 'cycles': A list of cycle numbers.
                  - 'dch_cap', 'ch_cap': Lists of discharge/charge capacities per cycle.
                  - 'HIs_dch', 'HIs_ch': Dictionaries of health indicators for each cycle.
        """
        alldata = pd.read_csv(file_path, header=None)
        alldata.columns = ['DataPoint', 'Cycle', 'Step Type', 'Time','Total Time', 'Current', 'Voltage',
                           'Capacity', 'Temperature']

        resample_interval = int(10)
        alldata = alldata.iloc[::resample_interval, :]

        # Calculate and filter dT/dV
        alldata['dT/dV'] = alldata.groupby('Cycle')['Temperature'].diff() / alldata.groupby('Cycle')['Voltage'].diff()
        alldata['dT/dV'] = alldata.groupby('Cycle')['dT/dV'].transform(self.filter_dT_dV)
        alldata['-dT/dV'] = -1 * alldata['dT/dV']
        alldata['Time'] = pd.to_timedelta(alldata['Time']).dt.total_seconds()

        # Handle infinite values and fill NaNs
        alldata = alldata.replace([np.inf, -np.inf], np.nan)
        alldata = alldata.bfill()

        # Separate charge and discharge data
        data_dch = alldata.loc[alldata['Step Type'] == 'CC DChg'].copy()
        data_ch = alldata.loc[alldata['Step Type'] == 'CC Chg'].copy()

        cycles = sorted(alldata['Cycle'].unique())[:-1]

        # --- Initialize data storage lists and dictionaries ---
        dch_cap, ch_cap = [], []

        # Time-series data
        time_dch, voltage_dch, current_dch, temperature_dch, DTV_dch = [], [], [], [], []
        time_ch, voltage_ch, current_ch, temperature_ch, DTV_ch = [], [], [], [], []

        # Health Indicator dictionaries
        HIs_dch = {
            'discharge_time': [], 'temperature_std': [], 'voltage_std': [],
            'dv_dt_mean': [], 'dv_dt_std': [], 'current_stability': [],
            'mean_temp': [], 'max_temp': [], 'voltage_integral': [], 'dv_dt_min': [],
        }
        HIs_ch = {
            'charge_time': [], 'voltage_plateau_time': [], 'voltage_at_80_soc': [],
            'temperature_std': [], 'voltage_std': [], 'dv_dt_mean': [], 'dv_dt_std': [],
            'current_stability': [], 'mean_temp': [], 'max_temp': [], 'voltage_integral': []
        }

        # --- Loop through cycles to extract data and calculate HIs ---
        for cycle in cycles:
            if cycle not in []:
                cycle_data_dch = data_dch[data_dch['Cycle'] == cycle]
                cycle_data_ch = data_ch[data_ch['Cycle'] == cycle]

                # Append time-series data
                time_dch.append(cycle_data_dch['Time'].values)
                voltage_dch.append(cycle_data_dch['Voltage'].values)
                current_dch.append(cycle_data_dch['Current'].values)
                temperature_dch.append(cycle_data_dch['Temperature'].values)
                DTV_dch.append(cycle_data_dch['-dT/dV'].values)

                time_ch.append(cycle_data_ch['Time'].values)
                voltage_ch.append(cycle_data_ch['Voltage'].values)
                current_ch.append(cycle_data_ch['Current'].values)
                temperature_ch.append(cycle_data_ch['Temperature'].values)
                DTV_ch.append(cycle_data_ch['-dT/dV'].values)

                # Append capacity
                dch_cap.append(np.max(cycle_data_dch['Capacity'].dropna()))
                ch_cap.append(np.max(cycle_data_ch['Capacity'].dropna()))

                # --- Calculate and Store Health Indicators (HIs) ---

                # dv/dt calculation
                dvdt_dch = np.diff(cycle_data_dch['Voltage'].values) / np.diff(cycle_data_dch['Time'].values)
                dvdt_ch = np.diff(cycle_data_ch['Voltage'].values) / np.diff(cycle_data_ch['Time'].values)

                # Discharge HIs
                HIs_dch['discharge_time'].append(cycle_data_dch['Time'].values[-1] - cycle_data_dch['Time'].values[0])
                HIs_dch['temperature_std'].append(np.std(cycle_data_dch['Temperature'].values))
                HIs_dch['voltage_std'].append(np.std(cycle_data_dch['Voltage'].values))
                HIs_dch['dv_dt_mean'].append(np.nanmean(dvdt_dch))
                HIs_dch['dv_dt_std'].append(np.nanstd(dvdt_dch))
                HIs_dch['dv_dt_min'].append(np.nanmin(dvdt_dch))
                HIs_dch['current_stability'].append(np.std(cycle_data_dch['Current'].values))
                HIs_dch['mean_temp'].append(np.mean(cycle_data_dch['Temperature'].values))
                HIs_dch['max_temp'].append(np.max(cycle_data_dch['Temperature'].values))
                
                try:
                    HIs_dch['voltage_integral'].append(np.trapz(cycle_data_dch['Voltage'].values, cycle_data_dch['Time'].values))
                except:
                    HIs_dch['voltage_integral'].append(np.nan)

                # Charge HIs
                HIs_ch['charge_time'].append(cycle_data_ch['Time'].values[-1] - cycle_data_ch['Time'].values[0])
                HIs_ch['temperature_std'].append(np.std(cycle_data_ch['Temperature'].values))
                HIs_ch['voltage_std'].append(np.std(cycle_data_ch['Voltage'].values))
                HIs_ch['dv_dt_mean'].append(np.nanmean(dvdt_ch))
                HIs_ch['dv_dt_std'].append(np.nanstd(dvdt_ch))
                HIs_ch['current_stability'].append(np.std(cycle_data_ch['Current'].values))
                HIs_ch['mean_temp'].append(np.mean(cycle_data_ch['Temperature'].values))
                HIs_ch['max_temp'].append(np.max(cycle_data_ch['Temperature'].values))
                try:
                    HIs_ch['voltage_integral'].append(np.trapz(cycle_data_ch['Voltage'].values, cycle_data_ch['Time'].values))
                except:
                    HIs_ch['voltage_integral'].append(np.nan)

                # Time to voltage plateau (e.g., 4.2 V during charge)
                plateau_voltage = 4.2
                above_plateau_idx = np.where(cycle_data_ch['Voltage'].values >= plateau_voltage)[0]
                if len(above_plateau_idx) > 0:
                    plateau_time = cycle_data_ch['Time'].values[above_plateau_idx[0]] - cycle_data_ch['Time'].values[0]
                else:
                    plateau_time = np.nan
                HIs_ch['voltage_plateau_time'].append(plateau_time)

                # Voltage at 80% SoC
                try:
                    cap_arr = cycle_data_ch['Capacity'].values
                    max_cap = np.nanmax(cap_arr)
                    cap_80 = 0.8 * max_cap
                    idx_80 = np.argmin(np.abs(cap_arr - cap_80))
                    HIs_ch['voltage_at_80_soc'].append(cycle_data_ch['Voltage'].values[idx_80])
                except:
                    HIs_ch['voltage_at_80_soc'].append(np.nan)

        # --- Package original (unequalized) time-series data ---
        data_original_dch = {
            'voltage': voltage_dch, 'temperature': temperature_dch,
            'time': time_dch, 'DTV': DTV_dch
        }
        data_original_ch = {
            'voltage': voltage_ch, 'temperature': temperature_ch,
            'time': time_ch, 'DTV': DTV_ch
        }

        # --- Equalize lengths time-series data ---
        data_eq_ch = {key: self.length_equalizer(val) for key, val in data_original_ch.items()}
        data_eq_dch = {key: self.length_equalizer(val) for key, val in data_original_dch.items()}

        # --- Final data structure for return ---
        return_data = {
            'data_dch': data_dch,
            'data_ch': data_ch,
            'Original_dch': data_original_dch,
            'Original_ch': data_original_ch,
            'Equalized_dch': data_eq_dch,
            'Equalized_ch': data_eq_ch,
            'cycles': cycles,
            'dch_cap': dch_cap,
            'ch_cap': ch_cap,
            'HIs_dch': HIs_dch,
            'HIs_ch': HIs_ch
        }

        print(f"Cell data loaded from {file_path}")
        return return_data


class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def extract_axis_data(self, cycle_data, ax):
        if ax != 'Time':
            return cycle_data[ax].dropna()
        else:
            return cycle_data['Time'].dropna().to_numpy() - cycle_data['Time'].iloc[0]

    def plot_3D(self, ax1, ax2, ax3, start_cycle, end_cycle):
        cmap = plt.get_cmap('plasma')
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        for cycle in range(start_cycle, end_cycle):
            if cycle not in []:
                cycle_data = self.data[(self.data['Cycle'] == cycle)]

                Ax1 = self.extract_axis_data(cycle_data, ax1)
                Ax2 = self.extract_axis_data(cycle_data, ax2)
                Ax3 = self.extract_axis_data(cycle_data, ax3)

                color = cmap(cycle / (end_cycle - 1))
                rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'
                num_ticks = 5
                tick_values = np.linspace(start_cycle, end_cycle, num_ticks + 2).astype(int)

                trace = go.Scatter3d(
                    x=Ax1,
                    y=Ax2,
                    z=Ax3,
                    mode='lines',
                    showlegend=False,
                    line=dict(
                        color=rgb_color,
                        colorbar=dict(
                            title='Cycle Number',
                            tickvals=tick_values,
                            ticks='inside'),
                        cmin=start_cycle,
                        cmax=end_cycle
                    ))
                fig.add_trace(trace)

        fig.update_layout(
            scene=dict(
                xaxis_title=ax1,
                yaxis_title=ax2,
                zaxis_title=ax3
            ))
        fig.show()

    def plot_degradation(self, mode, C_initial=None, marker='o', color='green', markersize=3, label=None):
        # mode : ['dch_cap'/'ch_cap', start cycle, reference capacity cycle]
        ch_dch = mode[0]
        start_cycle = mode[1]
        ref_cycle = mode[2]

        Capacity = self.data[ch_dch]

        if C_initial is not None:
            C_n = Capacity / C_initial
        else:
            C_n = Capacity / Capacity[ref_cycle]

        cycles = self.data['cycles']

        plt.plot(cycles[start_cycle:-3], C_n[start_cycle:-3], marker=marker, color=color, markersize=markersize, label=label)

    def plot_all(self, cycles_to_plot=[10, 80, 130],
                                dch_his_to_plot=['discharge_time', 'max_temp', 'voltage_std'],
                                ch_his_to_plot=['charge_time', 'max_temp', 'voltage_std'],
                                normalize_his=False,
                                hi_xaxis='capacity'):
        """
        Generates a comprehensive grid of plots to visualize battery data and health indicators.

        Args:
            processed_data (dict): The dictionary returned by a data loading function.
            cycles_to_plot (list, optional): A list of cycle numbers to display in the profile plot.
            dch_his_to_plot (list, optional): A list of strings specifying which discharge HIs to plot.
            ch_his_to_plot (list, optional): A list of strings specifying which charge HIs to plot.
            normalize_his (bool, optional): If True, normalizes HI values to a [0, 1] scale.
            hi_xaxis (str, optional): The x-axis for HI plots. Can be 'cycle' or 'capacity'.
        """
        processed_data = self.data

        # --- Unpack the data ---
        data_dch = processed_data['data_dch']
        data_ch = processed_data['data_ch']
        org_ch = processed_data['Original_ch']
        org_dch = processed_data['Original_dch']
        cycles = np.array(processed_data['cycles'])
        dch_cap = abs(np.array(processed_data['dch_cap']))
        ch_cap = np.array(processed_data['ch_cap'])
        HIs_dch = processed_data['HIs_dch']
        HIs_ch = processed_data['HIs_ch']

        charge_cycles_x_axis = np.arange(1, len(ch_cap) + 1)

        # --- Dynamic Figure Layout ---
        num_hi_plots = len(dch_his_to_plot) + len(ch_his_to_plot)
        num_rows = 1 + math.ceil(num_hi_plots / 2)
        fig = plt.figure(figsize=(18, 7 * num_rows))
        gs = fig.add_gridspec(num_rows, 2)

        hi_plot_count = 0
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'Greys', 'Blues', 
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn']
        
        fig.suptitle('Battery Performance and Health Indicators Analysis', fontsize=20)

        # === Plot 1: Cycle Profiles ===   
        ax1 = fig.add_subplot(gs[0, 0])
        colors = plt.cm.plasma(np.linspace(0, 1, len(cycles_to_plot)))
        for i, cycle in enumerate(cycles_to_plot):
            cycle_dch_df = data_dch[data_dch['Cycle'] == cycle]
            plt_data_dch = np.diff(cycle_dch_df['Voltage'].values) / np.diff(cycle_dch_df['Time'].values)
            ax1.plot(cycle_dch_df['Time'], cycle_dch_df['Voltage'], color=colors[i], linestyle='-', label=f'V_dch (Cyc {cycle})')
            cycle_ch_df = data_ch[data_ch['Cycle'] == cycle]
            plt_data_ch = np.diff(cycle_ch_df['Voltage'].values) / np.diff(cycle_ch_df['Time'].values)
            ax1.plot(cycle_ch_df['Time'], cycle_ch_df['Voltage'], color=colors[i], linestyle='--')
        ax1.set_title('Voltage Profiles (Solid=DChg, Dashed=Chg)', fontsize=14)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Voltage (V)', fontsize=12)
        ax1.legend()
        ax1.grid(True)

        # === Plot 2: Capacity Degradation ===
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(cycles, dch_cap, 'o-', label='Discharge Capacity')
        ax2.plot(cycles, ch_cap, 's--', label='Charge Capacity')
        ax2.axhline(y=0.8*dch_cap[3], color='r', linestyle='--', label='EOL Threshold (1.4Ah)')
        ax2.set_title('Capacity Degradation Path', fontsize=14)
        ax2.set_xlabel('Cycle Number', fontsize=12)
        ax2.set_ylabel('Capacity (Ah)', fontsize=12)
        ax2.legend()
        ax2.grid(True)

        # === Dynamic HI Scatter Plots ===
        # Discharge HIs
        for hi_name in dch_his_to_plot:
            row, col = 1 + hi_plot_count // 2, hi_plot_count % 2
            ax_hi = fig.add_subplot(gs[row, col])
            
            if hi_name in HIs_dch:
                x_axis_dch = dch_cap if hi_xaxis == 'capacity' else cycles
                hi_data = np.array(HIs_dch[hi_name])
                if normalize_his:
                    min_val, max_val = hi_data.min(), hi_data.max()
                    hi_data = (hi_data - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(hi_data)
                
                cmap = cmaps[hi_plot_count % len(cmaps)]
                scatter = ax_hi.scatter(x_axis_dch, hi_data, c=cycles, cmap=cmap, s=15, alpha=0.8)
                fig.colorbar(scatter, ax=ax_hi, label='Cycle Number')
                ax_hi.set_title(f'Discharge HI: {hi_name}', fontsize=14)
                ax_hi.set_xlabel('Capacity (Ah)' if hi_xaxis == 'capacity' else 'Cycle Number', fontsize=12)
                ax_hi.set_ylabel('Normalized Value' if normalize_his else 'HI Value', fontsize=12)
                ax_hi.grid(True)
                if hi_xaxis == 'capacity':
                    ax_hi.invert_xaxis()
            hi_plot_count += 1
            
        # Charge HIs
        for hi_name in ch_his_to_plot:
            row, col = 1 + hi_plot_count // 2, hi_plot_count % 2
            ax_hi = fig.add_subplot(gs[row, col])

            if hi_name in HIs_ch:
                x_axis_ch = ch_cap if hi_xaxis == 'capacity' else charge_cycles_x_axis
                hi_data = np.array(HIs_ch[hi_name])
                if normalize_his:
                    min_val, max_val = hi_data.min(), hi_data.max()
                    hi_data = (hi_data - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(hi_data)
                
                cmap = cmaps[hi_plot_count % len(cmaps)]
                scatter = ax_hi.scatter(x_axis_ch, hi_data, c=charge_cycles_x_axis, cmap=cmap, s=15, alpha=0.8)
                fig.colorbar(scatter, ax=ax_hi, label='Cycle Number')
                ax_hi.set_title(f'Charge HI: {hi_name}', fontsize=14)
                ax_hi.set_xlabel('Capacity (Ah)' if hi_xaxis == 'capacity' else 'Cycle Number', fontsize=12)
                ax_hi.set_ylabel('Normalized Value' if normalize_his else 'HI Value', fontsize=12)
                ax_hi.grid(True)
                if hi_xaxis == 'capacity':
                    ax_hi.invert_xaxis()
            hi_plot_count += 1

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        return cycle_ch_df