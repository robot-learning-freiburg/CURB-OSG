#!/usr/bin/env python2
'''
Visualization demo for panoptic COCO sample_data

The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.
'''
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import cv2
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id

# whether from the PNG are used or new colors are generated
generate_new_colors = True

json_file = '/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_gt/robotcar-labeling-v1.0.coco.json'
segmentations_folder = '/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_gt/robotcar-labeling-v1.0.coco/'
img_folder = '/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_gt/test'
panoptic_coco_categories = './panoptic_coco_categories.json'

with open(json_file, 'r') as f:
    coco_d = json.load(f)

annotations = coco_d['annotations']
ann = [ann for ann in annotations if ann['image_id'] == '1547475913057882'][0]
# ann = np.random.choice(coco_d['annotations'])

# with open(panoptic_coco_categories, 'r') as f:
#     categories_list = json.load(f)
categories_list = coco_d['categories']
categegories = {category['id']: category for category in categories_list }
category_name2id = {category['name']: category['id'] for category in categories_list}
target_cats = [category_name2id['traffic sign'], category_name2id['traffic light']]
print(target_cats)

# find input img that correspond to the annotation
# img = None
# for image_info in coco_d['images']:
#     if image_info['id'] == ann['image_id']:
#         try:
#             img = np.array(
#                 Image.open(os.path.join(img_folder, image_info['file_name']))
#             )
#         except:
#             print("Undable to find correspoding input image.")
#         break

img = np.array(
    Image.open(os.path.join(img_folder, ann['file_name']))
)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)

segmentation = np.array(
    Image.open(os.path.join(segmentations_folder, ann['file_name'])),
    dtype=np.uint8
)
segmentation_id = rgb2id(segmentation)

target_segments = []
if generate_new_colors:
    segmentation[:, :, :] = 0
    color_generator = IdGenerator(categegories)
    for segment_info in ann['segments_info']:
        if not segment_info['category_id'] in target_cats:
            continue
        color = color_generator.get_color(segment_info['category_id'])
        mask = segmentation_id == segment_info['id']
        segmentation[mask] = np.array(color + [255])

# find segments boundaries
# boundaries = find_boundaries(segmentation_id, mode='thick')

# depict boundaries
# segmentation[boundaries] = np.array([0, 0, 0, 255])

black = np.zeros_like(segmentation)
black[:,:, 3] = 255
plt.figure()
plt.imshow(black, alpha=1.0)
plt.imshow(img, alpha=0.5)
plt.imshow(segmentation, alpha=0.7)
plt.axis('off')

# if img is None:
#     plt.figure()
#     plt.imshow(segmentation)
#     plt.axis('off')
# else:
#     plt.figure(figsize=(9, 5))
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.subplot(122)
#     plt.imshow(segmentation)
#     plt.axis('off')
#     plt.tight_layout()
# plt.show()
# plt.savefig('test.png')
plt.savefig('testt.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
