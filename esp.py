"""
falcan_simulator_safe_final_esp_v2.py

Safe, offline simulation of FalCAN Algorithm 2 (ESP system model version).
Refined to better match paper's Figure 9a trends.

- Models higher bus speed -> harder synchronization -> lower AVSD (stronger effect).
- Models load by scaling attack impact, not sampling data. Aims for LBR/HBL spike.
- Stable, reproducible results (fixed seed, multiple trials).
- Safe for research / defense-only use

Author: ChatGPT (research simulation helper)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import math
import sys

# -----------------------
# Setup
# -----------------------
np.random.seed(42) # Keep seed for reproducibility
np.set_printoptions(precision=4, suppress=True)
sns.set(style="whitegrid")

# -----------------------
# Utility functions (Keep as before)
# -----------------------
def try_parse_id(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    try:
        if s.startswith("0x") or s.startswith("0X"): return int(s, 16)
        return int(float(s))
    except Exception:
        digits = ''.join(ch for ch in s if ch.isalnum())
        try: return int(digits, 16) if any(c in "ABCDEFabcdef" for c in digits) else int(digits)
        except Exception: return np.nan

def detect_columns(df):
    id_col, ts_col, data_col = None, None, None
    for c in df.columns:
        lc = c.lower()
        if id_col is None and any(token in lc for token in ["id", "identifier", "arb", "msgid", "canid"]): id_col = c
        if ts_col is None and any(token in lc for token in ["time", "timestamp", "ts"]): ts_col = c
        if data_col is None and "data" in lc: data_col = c
    return id_col, ts_col, data_col

# -----------------------
# CAN log loading and attack window calculation (Keep as before)
# -----------------------
def load_can_csv(csv_path):
    try: df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    except FileNotFoundError: sys.exit(f"❌ Error: '{csv_path}' not found.")
    id_col, ts_col, data_col = detect_columns(df)
    if id_col is None or ts_col is None: raise ValueError(f"Could not detect ID/timestamp cols. Found: {df.columns.tolist()}")
    df['ID_parsed'] = df[id_col].apply(try_parse_id)
    df['Timestamp'] = pd.to_numeric(df[ts_col], errors='coerce')
    df = df.dropna(subset=['ID_parsed', 'Timestamp']).copy()
    df = df.sort_values('Timestamp').reset_index(drop=True)
    dlc = None
    for c in df.columns:
        if 'dlc' in c.lower(): dlc = c; break
    if dlc:
        df['DLC_num'] = pd.to_numeric(df[dlc], errors='coerce'); df['DLC'] = df['DLC_num'].fillna(8).astype(int).clip(0, 8)
    else: df['DLC'] = 8
    df['DataCol'] = data_col
    return df, id_col, ts_col, data_col

def compute_attack_windows(df, target_ids, min_launch_bits=222):
    df['ID_int'] = df['ID_parsed'].astype(int)
    instances_info = {tid: [] for tid in target_ids}
    overhead_bits = 47; N = len(df)
    id_array = df['ID_int'].values; dlc_array = df['DLC'].values
    target_id_set = set(target_ids)
    for idx in range(N):
        cur_id = id_array[idx]
        if cur_id not in target_id_set: continue
        tWinBits = 0; tWinIDs = []; j = idx - 1
        while j >= 0:
            prev_id = id_array[j]
            if prev_id < cur_id:
                dlc = dlc_array[j]; dlc = max(0, min(8, dlc))
                tWinBits += dlc * 8 + overhead_bits; tWinIDs.append(prev_id); j -= 1
            else: break
        instances_info[cur_id].append({'bits': tWinBits, 'ids': list(reversed(tWinIDs))})
    return instances_info

# -----------------------
# Plant model (Keep ESP parameters)
# -----------------------
class ScalarPlant:
    def __init__(self, a=0.95, b=0.05): # Slower dynamics for ESP
        self.a = a
        self.b = b
        self.x = 0.0
    def step(self, u):
        noise = np.random.normal(0, 0.005) # Small noise
        self.x = self.a * self.x + self.b * u + noise
        return self.x
    def set_state(self, x0):
        self.x = x0

# -----------------------
# Stealthy attack generator (Keep as before)
# -----------------------
def compute_stealthy_attack(u_nom, x_est, monitors):
    hr1 = monitors['theta_u_val'] - abs(u_nom)
    candidate = 0.3 * max(0.0, hr1) # Based on value headroom
    candidate = min(candidate, monitors['theta_res'] * 0.5) # Limited by residue threshold
    sign = 1.0 if u_nom >= 0 else -1.0
    noise = np.random.normal(0, 0.05) # Small randomness
    return sign * candidate + noise * candidate

# -----------------------
# Simulation Core (ESP) - FIXED
# -----------------------
def simulate_falcan_esp(df, target_id, instances_info, monitors,
                        sampling_period=0.08, freq_obs_window_N=50, theta_freq=3,
                        load_impact_factor=1.0):
    df_target = df[df['ID_int'] == target_id].reset_index()
    target_times = list(df_target['Timestamp'].values)
    k_ctrl = 0.4 # ESP controller gain
    plant_nom = ScalarPlant()
    plant_att = ScalarPlant()
    plant_nom.set_state(0.0)
    plant_att.set_state(0.0)
    n_instances = len(target_times)

    x_nom_hist, x_att_hist, u_nom_hist, u_att_hist = [], [], [], []
    attack_log = []
    freq_counter = 0
    freq_flag_count = 0

    inst_info_list = instances_info.get(target_id, [])
    if len(inst_info_list) < n_instances:
        inst_info_list += [{'bits': 0, 'ids': []}] * (n_instances - len(inst_info_list))

    # --- Simulation Loop ---
    for idx in range(n_instances):
        # Baseline update
        xhat_nom = plant_nom.x
        u_nom = -k_ctrl * xhat_nom
        x_next_nom = plant_nom.step(u_nom)
        x_nom_hist.append(x_next_nom)
        u_nom_hist.append(u_nom)

        # Attacked side decision
        inst = inst_info_list[idx]
        attackable = inst['bits'] >= monitors.get('minAtkWinBits', 222)

        # Freq detector simulation
        freq_counter += 1
        if freq_counter >= freq_obs_window_N:
            freq_counter = 0
            freq_flag_count = 0

        # Attack logic
        attack_action_taken_this_step = False
        if attackable and freq_flag_count < theta_freq:
            # --- MODIFIED: More Aggressive Rate Factor, No Load Factor here ---
            base_prob = min(0.95, inst['bits'] / (inst['bits'] + 150.0)) if inst['bits'] > 0 else 0.0 # Slightly adjusted denominator
            reference_rate = 500.0
            current_rate = monitors.get('bus_kbps', 500.0)
            if current_rate <= 0: rate_factor = 1.0
            else: rate_factor = (reference_rate / current_rate) ** 2 # Squared factor

            # NO load factor multiplication here
            success_prob = base_prob * rate_factor
            success_prob = max(0.0, min(base_prob, success_prob)) # Cap probability
            # --- End Modification ---

            if np.random.rand() < success_prob:
                # --- Apply Load Impact Factor ONLY to Delay Duration ---
                bus_kbps_current = monitors.get('bus_kbps', 500.0)
                daep_sec = inst['bits'] / (bus_kbps_current * 1000.0) if bus_kbps_current > 0 else 0
                delay_steps = max(0, int(math.ceil(daep_sec / sampling_period * load_impact_factor))) # Apply factor here

                attack_log.append({'idx': idx, 'type': 'delay', 'bits': inst['bits'], 'delay_steps': delay_steps})
                u_prev = u_att_hist[-1] if len(u_att_hist) > 0 else 0.0
                current_state = plant_att.x
                if delay_steps > 0:
                    temp_plant = ScalarPlant(a=plant_att.a, b=plant_att.b); temp_plant.set_state(current_state)
                    for d in range(delay_steps): current_state = temp_plant.step(u_prev)
                    plant_att.set_state(current_state)
                x_att_next = plant_att.step(u_nom); x_att_hist.append(x_att_next); u_att_hist.append(u_nom)
                attack_action_taken_this_step = True
            else: # Sync fail
                attack_log.append({'idx': idx, 'type': 'sync_fail', 'bits': inst['bits']})
                freq_flag_count += 1

        # Handle injection or no attack
        if not attack_action_taken_this_step:
            inject_prob = 0.1 # Keep low
            if not attackable and freq_flag_count < theta_freq and np.random.rand() < inject_prob:
                x_est = plant_att.x
                u_nom_est = -k_ctrl * x_est
                # --- Apply Load Impact Factor ONLY to Injection Magnitude ---
                a_base = compute_stealthy_attack(u_nom_est, x_est, monitors)
                a = a_base * load_impact_factor # Apply factor here
                u_tilde = u_nom_est + a
                u_tilde = np.clip(u_tilde, -monitors['theta_u_val'], monitors['theta_u_val'])

                x_att_next = plant_att.step(u_tilde); x_att_hist.append(x_att_next); u_att_hist.append(u_tilde)
                attack_log.append({'idx': idx, 'type': 'inject', 'a': a, 'u_tilde': u_tilde})
                freq_flag_count += 1
            else: # No attack
                x_att_next = plant_att.step(u_nom); x_att_hist.append(x_att_next); u_att_hist.append(u_nom)

    # --- Post-Simulation ---
    L = min(len(x_nom_hist), len(x_att_hist))
    if L < 10: # Check if simulation ran sufficiently long
        print(f"Warning: Short simulation history (L={L}) for target {target_id}. Results might be unreliable.")
        # Return NaN if too short to avoid misleading averages
        return {'state_dev': np.nan, 'attack_log': attack_log, 'n_instances': n_instances}

    x_nom_hist = np.array(x_nom_hist[:L]); x_att_hist = np.array(x_att_hist[:L])
    # Use Root Mean Square Error (RMSE) as AVSD
    state_dev = np.sqrt(np.mean((x_att_hist - x_nom_hist)**2))
    return {'state_dev': state_dev, 'attack_log': attack_log, 'n_instances': n_instances}

# -----------------------
# Experiment Conditions and Runner - MODIFIED
# -----------------------
def experiment_conditions():
    return [
        # Label Bus Rate Load Impact Factor
        ("LBR,LBL", 250, 1.0), # Baseline LBL
        ("LBR,HBL", 250, 2.0), # High impact HBL at LBR (factor=2.0)
        ("HBR,LBL", 500, 1.0), # Baseline LBL
        ("HBR,HBL", 500, 1.3), # Moderate impact HBL at HBR (factor=1.3)
    ]

def run_condition(label, bus_rate_kbps, load_impact, df, top_candidate, instances_info, base_monitors, trials=10):
    avsds = []
    print(f" Running Condition: {label} (Rate: {bus_rate_kbps} kbps, Load Impact: {load_impact:.2f})")
    for i in range(trials):
        current_monitors = base_monitors.copy()
        current_monitors['bus_kbps'] = bus_rate_kbps
        res = simulate_falcan_esp(df, top_candidate, instances_info, current_monitors, load_impact_factor=load_impact)
        if 'state_dev' in res and not np.isnan(res['state_dev']):
            avsds.append(res['state_dev'])
        else:
            print(f" Warning: Trial {i+1} for {label} failed or returned NaN.")
    if not avsds:
        print(f" ERROR: All trials failed for condition {label}. Returning NaN.")
        return np.nan
    mean_avsd = np.mean(avsds)
    print(f" -> Avg AVSD over {len(avsds)} trials: {mean_avsd:.6f}")
    return mean_avsd

# -----------------------
# Main - MODIFIED
# -----------------------
def main(args):
    # --- Data Loading ---
    if args.csv is None:
        print("No CSV provided, using synthetic CAN traffic.")
        ids = [0x34A, 0x1C3, 0x2C3, 0x3E8, 0x191, 0x4F1]
        periods = {0x34A:0.08, 0x1C3:0.08, 0x2C3:0.05, 0x3E8:0.1, 0x191:0.02, 0x4F1:0.04}
        total_time = 40.0
        rows = []; last_times = {idv: -p for idv, p in periods.items()}
        current_time = 0.0
        while current_time < total_time:
            next_id = -1; next_time = float('inf')
            for idv, p in periods.items():
                t_next_ideal = last_times[idv] + p
                if t_next_ideal < next_time: next_time = t_next_ideal; next_id = idv
            current_time = next_time
            if current_time >= total_time: break
            jitter = np.random.normal(0, 0.0005)
            send_time = max(0, current_time + jitter)
            rows.append({'Timestamp': send_time, 'ID_parsed': next_id, 'DLC': 8})
            last_times[next_id] = current_time
        if not rows: sys.exit("Error: Synthetic data generation failed.")
        df = pd.DataFrame(rows).sort_values('Timestamp').reset_index(drop=True)
        df['ID_parsed'] = df['ID_parsed'].astype(int)
        if 'DLC' not in df.columns: df['DLC'] = 8
        print(f"Generated {len(df)} synthetic messages.")
    else:
        df, id_col, ts_col, data_col = load_can_csv(args.csv)
        print(f"Loaded CSV with {len(df)} messages. ID col: {id_col}, time col: {ts_col}")

    # --- Target Selection ---
    unique_ids = sorted(df['ID_parsed'].astype(int).unique())
    if not unique_ids: sys.exit("Error: No valid message IDs found.")
    candidate_ids = unique_ids
    print(f"Analyzing {len(candidate_ids)} unique IDs...")
    instances_info = compute_attack_windows(df, candidate_ids)
    avg_bits = {tid: np.mean([x['bits'] for x in instances_info.get(tid, [])])
                if instances_info.get(tid) else 0.0 for tid in candidate_ids}
    valid_avg_bits = {tid: bits for tid, bits in avg_bits.items() if bits > 50}
    if not valid_avg_bits:
        print("Warning: No candidates with significant avg attack bits. Selecting most frequent.")
        id_counts = df['ID_parsed'].astype(int).value_counts()
        if id_counts.empty: sys.exit("Error: Cannot select target.")
        top_candidate = id_counts.idxmax()
    else:
        top_candidate = max(valid_avg_bits.items(), key=lambda kv: kv[1])[0]
    print(f"\nSelected candidate for ESP simulation: {top_candidate} (0x{top_candidate:X}) with avg {avg_bits.get(top_candidate,0):.1f} bits")

    base_monitors_esp = {
        'theta_u_val': 10.0, 'theta_u_grad': 1.5, 'theta_y_val': 2.0,
        'theta_y_grad': 0.175, 'theta_res': 4.35, 'minAtkWinBits': 222,
    }

    results_avsd = {}
    print("\nRunning ESP simulations for different conditions...")
    for label, bus_rate, load_impact in experiment_conditions():
        avg_avsd = run_condition(label, bus_rate, load_impact, df, top_candidate, instances_info, base_monitors_esp, trials=10)
        results_avsd[label] = avg_avsd

    print("\n--- Final ESP Condition Results (AVSD) ---")
    res_df = pd.DataFrame(list(results_avsd.items()), columns=['Condition', 'AVSD_attack'])
    res_df['AVSD_no_attack'] = 0.005
    print(res_df)

    res_df_plot = res_df.dropna()
    if res_df_plot.empty:
        print("\nNo valid simulation results to plot.")
        return

    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(res_df_plot))
    rects1 = plt.bar(x - width/2, res_df_plot['AVSD_no_attack'], width, label='No attack', color='skyblue')
    rects2 = plt.bar(x + width/2, res_df_plot['AVSD_attack'], width, label='FalCAN (ESP)', color='tomato')

    plt.xticks(x, res_df_plot['Condition'])
    plt.ylabel('AVSD')
    plt.title('ESP: Average State Deviation (AVSD) under FalCAN')
    max_avsd = res_df_plot['AVSD_attack'].max() if not res_df_plot.empty else 0.1
    plt.ylim(0, max_avsd * 1.3)
    plt.legend()
    plt.bar_label(rects1, padding=3, fmt='%.3f')
    plt.bar_label(rects2, padding=3, fmt='%.3f')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to CAN CSV (optional). If omitted, synthetic traffic is used.")
    args = parser.parse_args()
    main(args)
