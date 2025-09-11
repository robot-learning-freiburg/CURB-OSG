# %%
import os
import csv
import cv2
from curb_projection import CameraModel
import numpy as np
from tqdm import tqdm

dataset = "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-14-14-15-12-radar-oxford-10k/"
models_dir = "/workspaces/collaborative-scene-graphs/src/curb_projection/intrinsics"
ins_file = "gps/ins.csv"
images_dir = "stereo/left"
output_dir = "./test"

N = 100  # sample N images
seed = 42

csv_file = os.path.join(dataset, ins_file)
images_dir = os.path.join(dataset, images_dir)

# read csv file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    ins_data = list(reader)

stamps = np.array([float(row[0]) for row in ins_data])
delta_t = np.diff(np.array([stamp / 1e6 for stamp in stamps], dtype=float))
v = np.array([np.linalg.norm([float(v_) for v_ in row[9:12]]) for row in ins_data])
delta_d = v[1:] * delta_t
# cumulative distance
d = np.cumsum(delta_d)
d = np.concatenate([[0], d])

# sample uniformly by distance travelled
# rng = np.random.RandomState(seed)
# sample uniformly by distance travelled
# distance_samples = rng.random((N,)) * d.max()
# distance_samples.sort()
# ^ random sampling disabled in favor of uniform spacing for more diversity
# draw N samples uniformly spaced by distance (cut first and last 10m)
distance_samples = np.linspace(10.0, d[-1] - 10.0, N)

# find closest timestamps
sample_indices = np.searchsorted(d, distance_samples)
sample_stamps = stamps[sample_indices]

# read image file names
image_files = list(filter(lambda f: f.endswith('.png'), os.listdir(images_dir)))
image_files.sort()
image_stamps = np.array([int(f.split('.')[0]) for f in image_files])
sample_image_indices = np.searchsorted(image_stamps, sample_stamps) - 1

os.makedirs(output_dir, exist_ok=True)
assert len(os.listdir(output_dir)) == 0, "Output directory is not empty"

model = CameraModel(models_dir, images_dir)

if 'stereo' in images_dir:
    # Bayer GR for stereo cam
    demosaic_mode = cv2.COLOR_BayerGR2BGR
elif 'mono' in images_dir:
    # Bayer BG for mono
    demosaic_mode = cv2.COLOR_BayerBG2BGR
else:
    raise ValueError("Unknown camera type")

input_images = [image_files[i] for i in sample_image_indices]
for f in tqdm(input_images):
    f_in = os.path.join(images_dir, f)
    image = cv2.imread(f_in, cv2.IMREAD_UNCHANGED)
    image = cv2.demosaicing(image, demosaic_mode)
    image = model.undistort(image)
    f_out = os.path.join(output_dir, f)
    cv2.imwrite(f_out, image)


# create a nice plot
import matplotlib.pyplot as plt
yx = np.array([[float(row[5]), float(row[6])] for row in ins_data])
sample_yx = yx[sample_indices]

# plot the trajectory
plt.plot(yx[:, 1], yx[:, 0])
# and the samples
plt.scatter(sample_yx[:, 1], sample_yx[:, 0], c='r')
plt.gca().set_axis_off()
plt.legend(['Trajectory', 'Sampled images'], loc='upper right')
# plt.show()
plt.savefig('trajectory.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf', transparent=True)