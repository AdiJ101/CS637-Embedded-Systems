"""
falcan_simulator_safe_final.py

Safe, offline simulation of FalCAN Algorithm 2 — with both:
  (1) detailed single-run plant simulation and attack summary
  (2) multi-condition AVSD experiment (LBR/LBL/HBR/HBL)

Author: ChatGPT (research-only helper)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import math

np.set_printoptions(precision=4, suppress=True)
sns.set(style="whitegrid")

# -----------------------
# Utilities
# -----------------------
def try_parse_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        return int(float(s))
    except Exception:
        digits = ''.join(ch for ch in s if ch.isalnum())
        try:
            return int(digits, 16) if any(c in "ABCDEFabcdef" for c in digits) else int(digits)
        except Exception:
            return np.nan

def detect_columns(df):
    id_col = None
    ts_col = None
    data_col = None
    for c in df.columns:
        lc = c.lower()
        if id_col is None and any(token in lc for token in ["id", "identifier", "arb", "msgid", "canid"]):
            id_col = c
        if ts_col is None and any(token in lc for token in ["time", "timestamp", "ts"]):
            ts_col = c
        if data_col is None and "data" in lc:
            data_col = c
    return id_col, ts_col, data_col

# -----------------------
# CAN Loading and Window Analysis
# -----------------------
def load_can_csv(csv_path):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    id_col, ts_col, data_col = detect_columns(df)
    if id_col is None or ts_col is None:
        raise ValueError(f"Could not detect ID or timestamp columns. Columns: {df.columns.tolist()}")
    df['ID_parsed'] = df[id_col].apply(try_parse_id)
    df['Timestamp'] = pd.to_numeric(df[ts_col], errors='coerce')
    df = df.dropna(subset=['ID_parsed', 'Timestamp']).copy()
    df = df.sort_values('Timestamp').reset_index(drop=True)
    dlc = None
    for c in df.columns:
        if 'dlc' in c.lower():
            dlc = c
            break
    if dlc:
        df['DLC'] = pd.to_numeric(df[dlc], errors='coerce').fillna(0).astype(int)
    else:
        df['DLC'] = 8
    df['DataCol'] = data_col
    return df, id_col, ts_col, data_col

def compute_attack_windows(df, target_ids, min_launch_bits=222):
    df['ID_int'] = df['ID_parsed'].astype(int)
    instances_info = {tid: [] for tid in target_ids}
    overhead_bits = 47
    for idx, row in df.iterrows():
        cur_id = int(row['ID_int'])
        if cur_id not in target_ids:
            continue
        tWinBits = 0
        tWinIDs = []
        j = idx - 1
        while j >= 0:
            prev_id = int(df.at[j, 'ID_int'])
            if prev_id < cur_id:
                dlc = int(df.at[j, 'DLC'])
                tWinBits += dlc * 8 + overhead_bits
                tWinIDs.append(prev_id)
                j -= 1
            else:
                break
        instances_info[cur_id].append({'bits': tWinBits, 'ids': list(reversed(tWinIDs))})
    return instances_info

# -----------------------
# Control and Plant Models
# -----------------------
def compute_stealthy_attack(u_nom, x_est, monitors):
    hr1 = monitors['theta_u_val'] - abs(u_nom)
    candidate = 0.3 * max(0.0, hr1)
    candidate = min(candidate, monitors['theta_res'])
    sign = 1.0 if u_nom >= 0 else -1.0
    return sign * candidate

class ScalarPlant:
    def __init__(self, a=0.9, b=0.1):
        self.a = a
        self.b = b
        self.x = 0.0
    def step(self, u):
        self.x = self.a * self.x + self.b * u
        return self.x
    def set_state(self, x0):
        self.x = x0

# -----------------------
# Simulation Core (Algo 2)
# -----------------------
def simulate_falcan_decisions(df, target_id, instances_info,
                              monitors,
                              sampling_period=0.08,
                              freq_obs_window_N=50,
                              theta_freq=3):
    df_target = df[df['ID_int'] == target_id].reset_index()
    target_times = list(df_target['Timestamp'].values)
    k_ctrl = 0.5
    plant_nom = ScalarPlant()
    plant_att = ScalarPlant()
    plant_nom.set_state(0.0)
    plant_att.set_state(0.0)
    n_instances = len(target_times)

    u_nom_history, u_att_history, x_nom_hist, x_att_hist = [], [], [], []
    attack_log, freq_flag_count, freq_counter = [], 0, 0
    inst_info_list = instances_info.get(target_id, [])
    if len(inst_info_list) < n_instances:
        inst_info_list += [{'bits': 0, 'ids': []}] * (n_instances - len(inst_info_list))

    for idx in range(n_instances):
        xhat_nom = plant_nom.x
        u_nom = -k_ctrl * xhat_nom
        x_next_nom = plant_nom.step(u_nom)
        x_nom_hist.append(x_next_nom)
        u_nom_history.append(u_nom)

        inst = inst_info_list[idx]
        attackable = inst['bits'] >= monitors.get('minAtkWinBits', 222)
        freq_counter += 1
        if freq_counter >= freq_obs_window_N:
            freq_counter = 0
            freq_flag_count = 0

        if attackable and freq_flag_count < theta_freq:
            success_prob = min(0.95, (inst['bits'] / (inst['bits'] + 200.0)) *
                               monitors.get('attack_scale', 1.0))
            if np.random.rand() < success_prob:
                daep_sec = inst['bits'] / (monitors.get('bus_kbps', 500) * 1000.0)
                delay_steps = int(math.ceil(daep_sec / sampling_period))
                u_prev = u_att_history[-1] if len(u_att_history) > 0 else 0.0
                for _ in range(delay_steps):
                    x_d = plant_att.step(u_prev)
                    x_att_hist.append(x_d)
                    u_att_history.append(u_prev)
                attack_log.append({'idx': idx, 'type': 'delay', 'bits': inst['bits']})
            else:
                attack_log.append({'idx': idx, 'type': 'sync_fail', 'bits': inst['bits']})
            freq_flag_count += 1
        else:
            inject_prob = 0.15
            if np.random.rand() < inject_prob:
                x_est = plant_att.x
                u_nom_est = -k_ctrl * x_est
                a = compute_stealthy_attack(u_nom_est, x_est, monitors)
                u_tilde = u_nom_est + a
                x_att_next = plant_att.step(u_tilde)
                x_att_hist.append(x_att_next)
                u_att_history.append(u_tilde)
                attack_log.append({'idx': idx, 'type': 'inject', 'a': a, 'u_tilde': u_tilde})
            else:
                x_att_next = plant_att.step(u_nom)
                x_att_hist.append(x_att_next)
                u_att_history.append(u_nom)

        if len(x_att_hist) < len(x_nom_hist):
            x_att_next = plant_att.step(u_nom)
            x_att_hist.append(x_att_next)
            u_att_history.append(u_nom)

    L = min(len(x_nom_hist), len(x_att_hist))
    x_nom_hist = x_nom_hist[:L]
    x_att_hist = x_att_hist[:L]
    state_dev = np.sqrt(np.mean((np.array(x_att_hist) - np.array(x_nom_hist))**2))
    return {
        'x_nom': np.array(x_nom_hist),
        'x_att': np.array(x_att_hist),
        'state_dev': state_dev,
        'attack_log': attack_log,
        'n_instances': n_instances
    }

# -----------------------
# Multi-condition AVSD experiment
# -----------------------
def experiment_conditions():
    return [
        ("LBR,LBL", 125, 0.5),
        ("LBR,HBL", 125, 1.0),
        ("HBR,LBL", 500, 0.5),
        ("HBR,HBL", 500, 1.0),
    ]

def run_condition(label, bus_rate_kbps, load_factor, df_template, top_candidate, trials=3):
    avsds = []
    for _ in range(trials):
        df = df_template.sample(frac=min(1.0, load_factor)).sort_values('Timestamp').reset_index(drop=True)
        monitors = {
            'theta_u_val': 30.0, 'theta_u_grad': 10.0,
            'theta_y_val': 25.0, 'theta_y_grad': 30.0,
            'theta_res': 4.35, 'minAtkWinBits': 222,
            'bus_kbps': bus_rate_kbps, 'attack_scale': load_factor,
        }
        inst_info = compute_attack_windows(df, [top_candidate])
        res = simulate_falcan_decisions(df, top_candidate, inst_info, monitors)
        avsds.append(res['state_dev'])
    return np.mean(avsds)

# -----------------------
# Main execution
# -----------------------
def main(args):
    if args.csv is None:
        print("No CSV provided, generating synthetic CAN traffic for demonstration.")
        ids = [0x34A, 0x1C3, 0x2C3, 0x3E8]
        total_time = 20.0
        rows, t = [], 0.0
        while t < total_time:
            for idv in ids:
                jitter = np.random.normal(0, 0.002)
                rows.append({'Timestamp': t + jitter, 'ID_parsed': idv, 'DLC': 8})
            t += 0.01
        df = pd.DataFrame(rows).sort_values('Timestamp').reset_index(drop=True)
    else:
        df, id_col, ts_col, data_col = load_can_csv(args.csv)
        print(f"Loaded CSV with {len(df)} messages. ID col: {id_col}, time col: {ts_col}")

    candidate_ids = sorted(df['ID_parsed'].astype(int).unique())[:8]
    instances_info = compute_attack_windows(df, candidate_ids)
    avg_bits = {tid: np.mean([x['bits'] for x in instances_info.get(tid, [])]) if len(instances_info.get(tid, []))>0 else 0.0 for tid in candidate_ids}
    print("Candidates avg window bits:", avg_bits)
    top_candidate = max(avg_bits.items(), key=lambda kv: kv[1])[0]
    print(f"Selected candidate: {top_candidate} (0x{top_candidate:X})")

    # Run single simulation (for printed info + trajectory plot)
    monitors = {
        'theta_u_val': 30.0, 'theta_u_grad': 10.0,
        'theta_y_val': 25.0, 'theta_y_grad': 30.0,
        'theta_res': 4.35, 'minAtkWinBits': 222,
        'bus_kbps': 500.0, 'attack_scale': 1.0,
    }
    sim_res = simulate_falcan_decisions(df, top_candidate, instances_info, monitors)

    print(f"\nSimulation complete. Instances processed: {sim_res['n_instances']}")
    print(f"Average state deviation (L2): {sim_res['state_dev']:.6f}")
    print(f"Attack events logged: {len(sim_res['attack_log'])}")
    breakdown = defaultdict(int)
    for ev in sim_res['attack_log']:
        breakdown[ev['type']] += 1
    print("Attack event breakdown:", dict(breakdown))

    # Plot time-series of plant states
    t_axis = np.arange(len(sim_res['x_nom'])) * 0.08
    plt.figure(figsize=(10,5))
    plt.plot(t_axis, sim_res['x_nom'], label='No attack (nominal)', linewidth=2)
    plt.plot(t_axis, sim_res['x_att'], label='Under simulated attack', linewidth=2, alpha=0.8)
    plt.xlabel('Time (samples)')
    plt.ylabel('Plant state')
    plt.title(f'Plant trajectories: nominal vs FalCAN (target 0x{top_candidate:X})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Multi-condition AVSD experiment ----
    base_df = df.copy()
    results = []
    for label, bus_rate, load in experiment_conditions():
        avsd_attack = run_condition(label, bus_rate, load, base_df, top_candidate)
        plant_nom = ScalarPlant()
        x_nom_hist = []
        for _ in range(200):
            u = np.random.normal(0, 0.05)
            x_nom_hist.append(plant_nom.step(u))
        avsd_no_attack = np.sqrt(np.mean((np.array(x_nom_hist))**2))
        results.append({'Condition': label, 'AVSD_attack': avsd_attack, 'AVSD_no_attack': avsd_no_attack})

    res_df = pd.DataFrame(results)
    print("\n--- Condition Results (AVSD) ---")
    print(res_df)

    plt.figure(figsize=(8,5))
    width = 0.35
    x = np.arange(len(res_df))
    plt.bar(x - width/2, res_df['AVSD_no_attack'], width, label='Under no attack', color='skyblue')
    plt.bar(x + width/2, res_df['AVSD_attack'], width, label='Under FalCAN', color='tomato')
    plt.xticks(x, res_df['Condition'])
    plt.ylabel('AVSD')
    plt.title('Average State Deviation (AVSD) under FalCAN')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to CAN CSV (optional)")
    args = parser.parse_args()
    main(args)
