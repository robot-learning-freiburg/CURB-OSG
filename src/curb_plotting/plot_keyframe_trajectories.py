#%%
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from hdl_graph_slam.msg import KeyframeArray_msg, Keyframe_msg
from geometry_msgs.msg import Pose
from collections import defaultdict
from draw_gt_trajectory import DATASET, read_gt_track

bags = """
/workspaces/collaborative-scene-graphs/src/curb_plotting/data/2025-02-20-no-lc.bag
/workspaces/collaborative-scene-graphs/src/curb_plotting/data/2025-02-21-false-lc.bag
/workspaces/collaborative-scene-graphs/src/curb_plotting/data/2025-02-19-keyframes-3agents-nicelc.bag
""".strip().split("\n")

def read_agent_trajectories(bag_path):
    bag = rosbag.Bag(bag_path, "r")
    keyframes = next(bag.read_messages(topics=["/optimized_keyframes"]))
    keyframes = keyframes.message.keyframes

    agent_trajectories = defaultdict(list)
    for keyframe in sorted(keyframes, key=lambda x: x.id):
        position = np.array([keyframe.odom.position.x, keyframe.odom.position.y])
        agent_trajectories[keyframe.agent_no].append(position)
    
    for agent_no, trajectory in agent_trajectories.items():
        trajectory = np.array(trajectory)
        # trajectory[:,1] = np.flip(trajectory[:,1])
        trajectory[:,1] = -trajectory[:,1]
        agent_trajectories[agent_no] = trajectory

    return agent_trajectories

def plot_bags(bags, titles):
    gt_track, _ = read_gt_track(DATASET)
    gt_track = np.array(gt_track)[:,:2]
    gt_track[:,1] = -gt_track[:,1]

    fig, axes = plt.subplots(1, len(bags), figsize=(len(bags) * 5, 5))

    for i, bag_path in enumerate(bags):
        agent_trajectories = read_agent_trajectories(bag_path)
        ax = axes[i]
        # decrease margins
        ax.margins(0.03)
        ax.plot(gt_track[:, 0], gt_track[:, 1], label="Ground Truth", color="black", linestyle="--")
        for agent_no, trajectory in agent_trajectories.items():
            trajectory = np.array(trajectory)
            colors = ["red", "blue", "green"]
            ax.plot(trajectory[:, 0], trajectory[:, 1], label=f"Agent {agent_no}", color=f"{colors[agent_no]}")
        
        # add little scale bar in meters
        scale_x, scale_y = ax.get_xlim(), ax.get_ylim()
        ax.plot([scale_x[1]-100, scale_x[1] - 300], [scale_y[0]+10, scale_y[0]+10], color="black", linewidth=3)
        ax.text(scale_x[1] - 200, scale_y[0]+40, "200m", color="black", ha="center")
        
        if i == 0:
            ax.legend(fontsize=16, loc="lower left")
        # disable axis labels
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(titles[i], y=-0.09, fontsize=23)

    plt.tight_layout()
    # save as pdf
    plt.savefig(f"./data/{' '.join(titles)}.pdf")

#%%
titles = [
    "(a) False Loop Closures",
    "(b) Correct Loop Closures",
]
plot_bags(bags[1:], titles)
plt.show()
# %%
titles = [
    "(a) No Loop Closures",
    "(b) False Loop Closures",
    "(c) Correct Loop Closures",
]
plot_bags(bags, titles)

# %%
