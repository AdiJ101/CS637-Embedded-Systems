import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --------------------------------------------
# 1. Load and inspect CAN log
# --------------------------------------------
csv_file = "SampleTwo.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    sys.exit(f"❌ Error: '{csv_file}' not found. Place it in the same folder as this script.")

print("✅ CSV loaded successfully.")
print("Columns detected:", df.columns.tolist())

# --------------------------------------------
# 2. Automatically detect ID and Timestamp columns
# --------------------------------------------
id_col = None
ts_col = None

for col in df.columns:
    lower = col.lower().strip()
    if any(x in lower for x in ["id", "identifier", "arb", "msgid"]):
        id_col = col
    if any(x in lower for x in ["time", "timestamp", "date"]):
        ts_col = col

if not id_col or not ts_col:
    sys.exit(f"❌ Could not automatically detect ID or timestamp columns.\n"
             f"Detected ID column: {id_col}\nDetected Timestamp column: {ts_col}")

print(f"🧩 Detected ID column: {id_col}")
print(f"🧩 Detected Timestamp column: {ts_col}")

# --------------------------------------------
# 3. Parse and clean data
# --------------------------------------------
def parse_can_id(x):
    try:
        if isinstance(x, str) and x.startswith("0x"):
            return int(x, 16)
        return int(x)
    except Exception:
        return np.nan

df["ID"] = df[id_col].apply(parse_can_id)
df["Timestamp"] = df[ts_col].astype(float)
df = df.dropna(subset=["ID", "Timestamp"]).sort_values(by="Timestamp")

# --------------------------------------------
# 4. Compute basic statistics
# --------------------------------------------
# Average period (mean inter-arrival time per ID)
df_period = df.groupby("ID")["Timestamp"].apply(lambda x: x.diff().dropna().mean()).reset_index()
df_period.rename(columns={"Timestamp": "AvgPeriod"}, inplace=True)

# Approx attack window (previous interval)
df["PrevTime"] = df.groupby("ID")["Timestamp"].shift(1)
df["AtkWinLen"] = (df["Timestamp"] - df["PrevTime"]).fillna(0)

# Average attack window per ID
df_aw = df.groupby("ID")["AtkWinLen"].mean().reset_index()
df_aw.rename(columns={"AtkWinLen": "AvgAtkWinLen"}, inplace=True)

# Data variance (if column present)
data_cols = [c for c in df.columns if "data" in c.lower()]
if data_cols:
    data_col = data_cols[0]
    print(f"🧩 Detected data column: {data_col}")
    def data_to_val(x):
        try:
            return int(str(x).replace(" ", "").replace("0x", ""), 16)
        except Exception:
            return 0
    df["DataVal"] = df[data_col].apply(data_to_val)
    df_var = df.groupby("ID")["DataVal"].var().reset_index()
    df_var.rename(columns={"DataVal": "DataVariance"}, inplace=True)
else:
    df_var = pd.DataFrame({"ID": df["ID"].unique(), "DataVariance": 0})

# Merge stats
stats = df_period.merge(df_aw, on="ID", how="outer").merge(df_var, on="ID", how="outer").fillna(0)

# --------------------------------------------
# 5. Normalize and compute Vulnerability Score
# --------------------------------------------
for col in ["AvgAtkWinLen", "AvgPeriod", "DataVariance"]:
    maxv, minv = stats[col].max(), stats[col].min()
    stats[f"{col}_norm"] = (stats[col] - minv) / (maxv - minv) if maxv != minv else 0.5

stats["VulnerabilityScore"] = (
    0.5 * stats["AvgAtkWinLen_norm"]
    + 0.3 * (1 - stats["AvgPeriod_norm"])
    + 0.2 * stats["DataVariance_norm"]
)

# --------------------------------------------
# 6. Identify top target
# --------------------------------------------
top_target = stats.loc[stats["VulnerabilityScore"].idxmax()]

print("\n--- 🔍 Top Target Message ---")
print(f"ID: {int(top_target['ID'])} (0x{int(top_target['ID']):X})")
print(f"Average Period: {top_target['AvgPeriod']:.6f} s")
print(f"Average Attack Window Length: {top_target['AvgAtkWinLen']:.6f} s")
print(f"Data Variance: {top_target['DataVariance']:.2f}")
print(f"Vulnerability Score: {top_target['VulnerabilityScore']:.3f}")

# --------------------------------------------
# 7. Visualization
# --------------------------------------------
sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(12, 6))
sns.barplot(x="ID", y="VulnerabilityScore", data=stats, palette="coolwarm")
plt.title("Vulnerability Score per Message ID")
plt.xlabel("CAN Message ID")
plt.ylabel("Vulnerability Score")
plt.axvline(x=list(stats["ID"]).index(top_target["ID"]), color="red", linestyle="--", label="Top Target")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=stats,
    x="AvgPeriod",
    y="AvgAtkWinLen",
    size="VulnerabilityScore",
    hue="VulnerabilityScore",
    palette="viridis",
    legend=True
)
plt.title("Attack Window vs Periodicity")
plt.xlabel("Average Period (s)")
plt.ylabel("Average Attack Window (s)")
plt.tight_layout()
plt.show()
