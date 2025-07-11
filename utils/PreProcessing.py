import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def compute_rul(soh_values, cycle_indices=None, threshold=0.8):
    soh_values = np.array(soh_values)

    if cycle_indices is None:
        cycle_indices = np.arange(len(soh_values))
    else:
        cycle_indices = np.array(cycle_indices)

    try:
        eol_index = np.where(soh_values <= threshold)[0][0]
        eol_cycle = cycle_indices[eol_index]
    except IndexError:
        eol_cycle = cycle_indices[-1] + 1

    rul = eol_cycle - cycle_indices
    rul = np.maximum(rul, 0)
    return rul

def remove_outliers(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mask = (data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)
    return data[mask], mask

def normalize(inputs_train, inputs_test):
    features_train = inputs_train[:, :-1]
    cycle_train = inputs_train[:, -1:]

    features_test = inputs_test[:, :-1]
    cycle_test = inputs_test[:, -1:]

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train.numpy())
    features_test_scaled = scaler.transform(features_test.numpy())

    inputs_train_scaled = tf.concat([
        tf.convert_to_tensor(features_train_scaled, dtype=tf.float32),
        cycle_train
    ], axis=1)

    inputs_test_scaled = tf.concat([
        tf.convert_to_tensor(features_test_scaled, dtype=tf.float32),
        cycle_test
    ], axis=1)

    return inputs_train_scaled, inputs_test_scaled, scaler

def remove_outliers_multihi(data_matrix, z_thresh=2):
    data_matrix = np.array(data_matrix)
    z_scores = np.abs((data_matrix - np.mean(data_matrix, axis=0)) / np.std(data_matrix, axis=0))
    mask = np.all(z_scores < z_thresh, axis=1)
    return mask

def process_cells(cell_ids, data_dict, C0, ch_dch='ch', selected_HIs_dch_list=None, selected_HIs_ch_list=None, remove_outliers=True, z_thresh=2):
    Cycle_list, Capacity_list, RUL_list, Temp_list, Volt_list = [], [], [], [], []
    agg_HIs_ch = {k: [] for k in selected_HIs_ch_list}
    agg_HIs_dch = {k: [] for k in selected_HIs_dch_list}

    for cond in cell_ids:
        cell_data = data_dict[cond]

        Capacity = cell_data[f'{ch_dch}_cap']
        Cycle = cell_data['cycles']
        Temp = cell_data[f'Equalized_{ch_dch}']['temperature']
        Volt = cell_data[f'Equalized_{ch_dch}']['voltage']

        RUL = compute_rul(Capacity, Cycle, threshold=0.7 * C0)

        # Stack selected HIs for outlier detection
        selected_dch_HIs = np.stack([cell_data['HIs_dch'][hi] for hi in selected_HIs_dch_list], axis=-1)
        selected_ch_HIs = np.stack([cell_data['HIs_ch'][hi] for hi in selected_HIs_ch_list], axis=-1)
        selected_HIs = np.concatenate([selected_dch_HIs, selected_ch_HIs], axis=-1)

        if remove_outliers:
            mask = remove_outliers_multihi(selected_HIs, z_thresh=z_thresh)
        else:
            mask = np.ones(selected_HIs.shape[0], dtype=bool)  # Keep all data

        # Filter everything using the mask
        Cycle = list(np.array(Cycle)[mask])
        Capacity = list(np.array(Capacity)[mask])
        RUL = list(np.array(RUL)[mask])
        Temp = list(np.array(Temp)[mask])
        Volt = list(np.array(Volt)[mask])

        for hi in selected_HIs_dch_list:
            filtered = list(np.array(cell_data['HIs_dch'][hi])[mask])
            agg_HIs_dch[hi].append(filtered)

        for hi in selected_HIs_ch_list:
            filtered = list(np.array(cell_data['HIs_ch'][hi])[mask])
            agg_HIs_ch[hi].append(filtered)

        Cycle_list.append(Cycle)
        Capacity_list.append(Capacity)
        RUL_list.append(RUL)
        Temp_list.append(Temp)
        Volt_list.append(Volt)

    # Flatten
    Cycles = [c for b in Cycle_list for c in b]
    Capacities = [c for b in Capacity_list for c in b]
    RULs = [r for b in RUL_list for r in b]
    Temps = [t for b in Temp_list for t in b]
    Volts = [v for b in Volt_list for v in b]

    HIs_ch = {name: [item for sublist in dat for item in sublist] for name, dat in agg_HIs_ch.items()}
    HIs_dch = {name: [item for sublist in dat for item in sublist] for name, dat in agg_HIs_dch.items()}

    # Convert to tensors
    Cycles = tf.expand_dims(tf.convert_to_tensor(Cycles, dtype=tf.float32), axis=-1)
    Capacities = tf.expand_dims(tf.convert_to_tensor(Capacities, dtype=tf.float32), axis=-1)
    RULs = tf.expand_dims(tf.convert_to_tensor(RULs, dtype=tf.float32), axis=-1)
    Temps = tf.convert_to_tensor(pad_sequences(Temps, dtype='float32', padding='post'))
    Volts = tf.convert_to_tensor(pad_sequences(Volts, dtype='float32', padding='post'))

    for k in HIs_ch:
        HIs_ch[k] = tf.expand_dims(tf.convert_to_tensor(HIs_ch[k], dtype=tf.float32), axis=-1)
    for k in HIs_dch:
        HIs_dch[k] = tf.expand_dims(tf.convert_to_tensor(HIs_dch[k], dtype=tf.float32), axis=-1)

    return Cycles, Capacities, RULs, Temps, Volts, HIs_ch, HIs_dch, RUL_list, Capacity_list, Cycle_list