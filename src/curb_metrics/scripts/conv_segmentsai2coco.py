# %%
import json
import os
import urllib.request
from tqdm import tqdm
import numpy as np

from curb_projection.utils import ColorGen

# GT_JSON = "/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_gt/robotcar-labeling-maryam-v0.43.json"
GT_JSON = "/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_gt/robotcar-labeling-v1.0.json"
COCO_JSON = GT_JSON.strip(".json") + ".coco.json"
PNG_PATH = GT_JSON.strip(".json") + ".coco"

os.makedirs(PNG_PATH, exist_ok=True)

print(f"JSON: {GT_JSON}")
print(f"PNG: {PNG_PATH}")


with open(GT_JSON, "r") as f:
    gt_json = json.load(f)

colorgen = ColorGen()

categories = gt_json["dataset"]["task_attributes"]["categories"]

for i in range(len(categories)):
    categories[i]["isthing"] = False
    rgb = colorgen.get_color(categories[i]["name"])
    categories[i]["color"] = list(int(n) for n in rgb*255)

annotations = []
for e in tqdm(gt_json["dataset"]["samples"]):
    gt = e["labels"]["ground-truth"]
    if gt is None:
        continue

    ann = {
        "file_name": e["name"],
        "image_id": e["name"].split(".")[0],
        "segments_info": gt["attributes"]["annotations"]
    }

    for i in range(len(ann["segments_info"])):
        ann["segments_info"][i]["iscrowd"] = 0

    png_out = os.path.join(PNG_PATH, e["name"])
    if not os.path.exists(png_out):
        urllib.request.urlretrieve(gt["attributes"]["segmentation_bitmap"]["url"], png_out)
        
    annotations.append(ann)

coco_json = {"categories": categories, "annotations": annotations}

with open(COCO_JSON, "w") as f:
    json.dump(coco_json, f)


# %%
