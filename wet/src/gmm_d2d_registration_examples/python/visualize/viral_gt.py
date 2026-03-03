import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
python visualize/viral_gt.py --dataset nya_01 --start-id 1250 --final-id 2000
python visualize/viral_gt.py --dataset eee_03 --start-id 1250 --final-id 1500
"""


GT_PATHS = {
    "eee_03": "/home/sonieth2/thesis/VIRAL/ntuviral_gt/eee_03/ground_truth.csv",
    "nya_01": "/home/sonieth2/thesis/VIRAL/ntuviral_gt/nya_01/ground_truth.csv",
}

COLORS = {
    "eee_03": "royalblue",
    "nya_01": "tomato",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["eee_03", "nya_01"], required=True)
parser.add_argument("--start-id", type=int, default=0, help="Start trajectory at this index (inclusive)")
parser.add_argument("--final-id", type=int, default=None, help="End trajectory at this index (inclusive)")
args = parser.parse_args()

df = pd.read_csv(GT_PATHS[args.dataset])
x = df["field.pose.position.x"].to_numpy()
y = df["field.pose.position.y"].to_numpy()
z = df["field.pose.position.z"].to_numpy()

end = args.final_id + 1 if args.final_id is not None else len(x)
x, y, z = x[args.start_id:end], y[args.start_id:end], z[args.start_id:end]

print(f"{args.dataset}: {len(x)} poses  [{args.start_id} : {end - 1}]  (total in file: {len(df)})")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

color = COLORS[args.dataset]
ax.plot(x, y, z, color=color, linewidth=1.2)
ax.scatter(x[0], y[0], z[0], color=color, marker="o", s=60, zorder=5, label="start")
ax.scatter(x[-1], y[-1], z[-1], color=color, marker="^", s=60, zorder=5, label="end")

title = f"VIRAL GT — {args.dataset}  [{args.start_id} : {end - 1}]"
ax.set_title(title + "\n● start  ▲ end")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
plt.tight_layout()
plt.show()
