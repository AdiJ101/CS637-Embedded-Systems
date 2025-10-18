"""
falcan_simulator_safe.py

Safe, offline simulation of the high-level decision logic from "Algo.2" (FalCAN),
for research/defense/HIL-style simulation only.

- Reads a CAN CSV (or uses synthetic traffic)
- Identifies a candidate target ID
- Computes attack windows (observation & launch windows) offline
- Simulates an event-triggered attack decision loop (no real CAN transmissions)
  * when attackable instance: model an attempted synchronization -> modeled delay
  * or inject a stealthy falsified control (computed to obey simple monitors)
- Simulates a simple LTI plant under both nominal and attacked inputs and
  shows the effect on state trajectories.

IMPORTANT: This is NOT attack code for real vehicles. It does not transmit frames,
does not contain synchronization/transmit logic, and is safe for educational use.

Dependencies:
  pip install numpy pandas matplotlib seaborn

Usage:
  python falcan_simulator_safe.py [--csv SampleTwo.csv]

Author: ChatGPT (safe research helper)
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
# Utilities and helpers
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
        # try to extract digits
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
# Core: analyze CAN log to compute attack windows
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
    # DLC if present
    dlc = None
    for c in df.columns:
        if 'dlc' in c.lower():
            dlc = c
            break
    if dlc:
        df['DLC'] = pd.to_numeric(df[dlc], errors='coerce').fillna(0).astype(int)
    else:
        df['DLC'] = 8  # assume 8 bytes if unknown
    df['DataCol'] = data_col
    return df, id_col, ts_col, data_col

def compute_attack_windows(df, target_ids, min_launch_bits=222):
    """
    For each target ID, analyze preceding messages to compute an 'attack observation window'
    and launch window length in bits (approximation using DLC*8 + protocol overhead).
    Returns per-ID: list of per-instance window lengths and the IDs composing the window.
    """
    # For quick lookups, convert IDs to ints
    df['ID_int'] = df['ID_parsed'].astype(int)
    instances_info = {tid: [] for tid in target_ids}

    # We'll iterate through traffic and for each occurrence of a target id, compute preceding consecutive
    # higher-priority messages (IDs with numeric value < target ID). We'll sum (DLC*8 + overhead) as bits.
    # This is analogous to tWinLen calculation in the paper.
    overhead_bits = 47  # approximate overhead used in your C code
    N = len(df)
    for idx, row in df.iterrows():
        cur_id = int(row['ID_int'])
        if cur_id not in target_ids:
            continue
        # walk backwards
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
# Attack-vector compute (safe, approximate)
# -----------------------
def compute_stealthy_attack(u_nom, x_est, monitors):
    """
    Compute a small stealthy perturbation 'a' to add to nominal control u_nom.
    This is a research-only, conservative routine that keeps a within provided monitor bounds.
    monitors: dict with keys theta_u_val, theta_u_grad, theta_y_val, theta_y_grad, theta_res
    For simplicity we return a scalar perturbation proportional to remaining headroom.
    """
    # Conservative rule: choose a fraction (e.g., 0.3) of the smallest headroom among constraints
    # headroom1 = theta_u_val - |u_nom|
    hr1 = monitors['theta_u_val'] - abs(u_nom)
    # gradient headroom (we don't have previous u here in simplistic model) use theta_u_grad
    hr2 = monitors['theta_u_grad']
    # use small fraction:
    candidate = 0.3 * max(0.0, hr1)
    # ensure not exceeding residual-based allowance (theta_res scaled)
    candidate = min(candidate, monitors['theta_res'])
    # sign choice: push in same direction as u_nom for amplification
    sign = 1.0 if u_nom >= 0 else -1.0
    return sign * candidate

# -----------------------
# Plant model and simulation
# -----------------------
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
# High-level event-driven ATTACKMETHOD simulation (analysis-only)
# -----------------------
def simulate_falcan_decisions(df, target_id, instances_info,
                              monitors,
                              sampling_period=0.08,
                              max_observation_windows=2,
                              freq_obs_window_N=50,
                              theta_freq=3):
    """
    Simulate the event-triggered attack decisions described in Algo.2 at a high level.
    - df: CAN traffic DataFrame
    - target_id: integer ID to target
    - instances_info: output from compute_attack_windows for this target
    - monitors: dict of monitoring thresholds
    - sampling_period: nominal controller sampling period (s)
    Returns:
      - results dict with baseline and attacked plant trajectories and stats
    """
    # Build list of timestamps where target messages occur
    df_target = df[df['ID_int'] == target_id].reset_index()
    target_times = list(df_target['Timestamp'].values)
    # Baseline: produce nominal control values (-Kx), here we use simple proportional controller
    # We'll model controller as u = -k * x_hat, with x_hat assumed = plant state (simplified)
    k_ctrl = 0.5
    # Initialize plants
    plant_nom = ScalarPlant()
    plant_att = ScalarPlant()
    plant_nom.set_state(0.0)
    plant_att.set_state(0.0)

    # Prepare time points: we step through each target instance (discrete-time controller)
    n_instances = len(target_times)
    u_nom_history = []
    u_att_history = []
    x_nom_hist = []
    x_att_hist = []
    attack_log = []
    freq_flag_count = 0
    freq_counter = 0

    # For simplicity, map instances_info list order to target_times order
    inst_info_list = instances_info.get(target_id, [])
    # pad if mismatched
    if len(inst_info_list) < n_instances:
        # pad with zeros
        inst_info_list = inst_info_list + [{'bits': 0, 'ids': []}] * (n_instances - len(inst_info_list))

    for idx in range(n_instances):
        # baseline control computation
        xhat_nom = plant_nom.x
        u_nom = -k_ctrl * xhat_nom
        # apply to nominal plant
        x_next_nom = plant_nom.step(u_nom)
        x_nom_hist.append(x_next_nom)
        u_nom_history.append(u_nom)

        # attacked side: decide whether to attempt attack at this instance
        inst = inst_info_list[idx]
        attackable = inst['bits'] >= monitors.get('minAtkWinBits', 222)
        will_attack = False
        attacked_by_delay = False
        injected_u = None

        # frequency-based detector: count occurrences in recent window (simplified)
        freq_counter += 1
        if freq_counter >= freq_obs_window_N:
            freq_counter = 0
            # reset flag count (mimic paper's behavior)
            freq_flag_count = 0

        # if attackable and not suppressed by freq detector, we decide action
        if attackable and freq_flag_count < theta_freq:
            # model an attempt to sync (we do NOT implement actual sync). We'll model successful synchronization
            # with some probability (platform uncertainty). Here set a success_prob derived from window size.
            success_prob = min(0.95, inst['bits'] / (inst['bits'] + 200.0)) if inst['bits'] > 0 else 0.0
            if np.random.rand() < success_prob:
                # successful synchronization -> delay the true control for a modeled duration (daep)
                # model delay daep in seconds as proportional to bits (bits / bus_kbps)
                daep_sec = inst['bits'] / (monitors.get('bus_kbps', 500) * 1000.0)  # bits / bps
                # compute how many sampling intervals this delays
                delay_steps = int(math.ceil(daep_sec / sampling_period))
                # effect: plant is actuated with previous u for extra delay_steps steps, then nominal u resumes
                attacked_by_delay = True
                will_attack = True
                attack_log.append({'idx': idx, 'type': 'delay', 'bits': inst['bits'], 'delay_steps': delay_steps})
                # implement delay by applying u_prev multiple times on attacked plant
                # if first instance, previous u is zero
                u_prev = u_att_history[-1] if len(u_att_history) > 0 else 0.0
                for d in range(delay_steps):
                    x_d = plant_att.step(u_prev)
                    x_att_hist.append(x_d)
                    u_att_history.append(u_prev)
            else:
                # synchronization failed -> no action
                attack_log.append({'idx': idx, 'type': 'sync_fail', 'bits': inst['bits']})
                will_attack = False
            # increment frequency flag if we attempted sync
            freq_flag_count += 1
        else:
            # either not attackable or freq detector suppressed: optionally inject falsified control AFTER true control
            # We'll choose to inject a stealthy falsified control u_tilde with small amplitude computed by compute_stealthy_attack
            # Design attacker to sometimes inject with small probability (simulate strategy)
            inject_prob = 0.15  # small
            if np.random.rand() < inject_prob:
                # craft u_tilde
                # estimate x for attacker (simplified as plant_att.x)
                x_est = plant_att.x
                u_nom_est = -k_ctrl * x_est
                a = compute_stealthy_attack(u_nom_est, x_est, monitors)
                u_tilde = u_nom_est + a
                # apply u_tilde for one step then resume
                x_att_next = plant_att.step(u_tilde)
                x_att_hist.append(x_att_next)
                u_att_history.append(u_tilde)
                attack_log.append({'idx': idx, 'type': 'inject', 'a': a, 'u_tilde': u_tilde})
                will_attack = True
            else:
                # no attack -> normal control applied
                x_att_next = plant_att.step(u_nom)
                x_att_hist.append(x_att_next)
                u_att_history.append(u_nom)

        # if we didn't already push an attacked action (e.g., delay loop already advanced), ensure alignment
        # Align lengths: baseline progressed by 1 (x_nom_hist appended), attacked side might have appended >=1 steps.
        # If attacked side appended exactly one step we are good; if 0, apply nominal step
        if len(x_att_hist) < len(x_nom_hist):
            # sync up with nominal steps
            x_att_next = plant_att.step(u_nom)
            x_att_hist.append(x_att_next)
            u_att_history.append(u_nom)

    # Trim histories to equal lengths
    L = min(len(x_nom_hist), len(x_att_hist))
    x_nom_hist = x_nom_hist[:L]
    x_att_hist = x_att_hist[:L]
    u_nom_history = u_nom_history[:L]
    u_att_history = u_att_history[:L]

    # compute average state deviation metric (L2)
    state_dev = np.sqrt(np.mean((np.array(x_att_hist) - np.array(x_nom_hist))**2))

    results = {
        'x_nom': np.array(x_nom_hist),
        'x_att': np.array(x_att_hist),
        'u_nom': np.array(u_nom_history),
        'u_att': np.array(u_att_history),
        'attack_log': attack_log,
        'state_dev': state_dev,
        'n_instances': n_instances
    }
    return results

# -----------------------
# Top-level run
# -----------------------
def main(args):
    if args.csv is None:
        print("No CSV provided, generating synthetic CAN traffic for demonstration.")
        # create synthetic traffic DataFrame
        # We'll generate timestamps and IDs; choose several IDs and periodicities
        ids = [0x34A, 0x1C3, 0x2C3, 0x3E8]  # sample IDs
        periods = {0x34A:0.08, 0x1C3:0.08, 0x2C3:0.05, 0x3E8:0.1}
        total_time = 20.0
        rows = []
        t = 0.0
        while t < total_time:
            for idv in ids:
                # occasional jitter
                jitter = np.random.normal(0, 0.002)
                rows.append({'Timestamp': t + jitter, 'ID_parsed': idv, 'DLC': 8})
            t += 0.01
        df = pd.DataFrame(rows).sort_values('Timestamp').reset_index(drop=True)
    else:
        df, id_col, ts_col, data_col = load_can_csv(args.csv)
        print(f"Loaded CSV with {len(df)} messages. ID col detected: {id_col}, time col: {ts_col}, data col: {data_col}")

    # choose candidate target IDs to analyze: either from args or all unique IDs
    unique_ids = sorted(df['ID_parsed'].astype(int).unique())
    # For demo, pick a subset if many:
    if len(unique_ids) > 8:
        candidate_ids = unique_ids[:8]
    else:
        candidate_ids = unique_ids

    # compute attack windows for candidates
    instances_info = compute_attack_windows(df, candidate_ids, min_launch_bits=222)

    # Score candidates by average bits (simple heuristic)
    avg_bits = {tid: np.mean([x['bits'] for x in instances_info.get(tid, [])]) if len(instances_info.get(tid, []))>0 else 0.0 for tid in candidate_ids}
    # select top candidate
    top_candidate = max(avg_bits.items(), key=lambda kv: kv[1])[0]
    print("Candidates avg window bits:", avg_bits)
    print(f"Selected candidate (highest avg bits): {top_candidate} (0x{top_candidate:X})")

    # set monitors (example conservative thresholds)
    monitors = {
        'theta_u_val': 30.0,   # control value range (abs)
        'theta_u_grad': 10.0,  # per-sample gradient allowed
        'theta_y_val': 25.0,
        'theta_y_grad': 30.0,
        'theta_res': 4.35,
        'minAtkWinBits': 222,
        'bus_kbps': 500.0
    }

    # run simulation
    sim_res = simulate_falcan_decisions(df, top_candidate, instances_info, monitors,
                                        sampling_period=0.08, freq_obs_window_N=50, theta_freq=3)

    print(f"\nSimulation complete. Instances processed: {sim_res['n_instances']}")
    print(f"Average state deviation (L2) due to simulated attacks: {sim_res['state_dev']:.6f}")
    print(f"Number of attack events logged: {len(sim_res['attack_log'])}")

    # show attack log summary
    type_counts = defaultdict(int)
    for ev in sim_res['attack_log']:
        type_counts[ev['type']] += 1
    print("Attack event breakdown:", dict(type_counts))

    # Plot results
    t = np.arange(len(sim_res['x_nom'])) * 0.08
    plt.figure(figsize=(10,5))
    plt.plot(t, sim_res['x_nom'], label='No attack (nominal)', linewidth=2)
    plt.plot(t, sim_res['x_att'], label='Under simulated attack', linewidth=2, alpha=0.8)
    plt.xlabel('Sample index (approx)')
    plt.ylabel('Plant state (scalar)')
    plt.title(f'Plant state: nominal vs simulated FalCAN decisions (target 0x{top_candidate:X})')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to CAN CSV (optional). If omitted, synthetic traffic is used.")
    args = parser.parse_args()
    main(args)
