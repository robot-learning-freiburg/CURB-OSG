#!/bin/bash

# script to download models for OpenGraph pipeline
# according to https://github.com/BIT-DYN/OpenGraph/tree/master?tab=readme-ov-file

target_dir = /workspaces/collaborative-scene-graphs/src/curb_projection/saved_models

tag2text_model="tag2text_swin_14m.pth"
tag2text_url="https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth"

if [ ! -f "$target_dir/$tag2text_model" ]; then
    echo "Downloading $tag2text_model to $target_dir/$tag2text_model"
    wget -O "$target_dir/$tag2text_model" "$tag2text_url"
fi


gdino_model="groundingdino_swint_ogc.pth"
gdino_url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

if [ ! -f "$target_dir/$gdino_model" ]; then
    echo "Downloading $gdino_model to $target_dir/$gdino_model"
    wget -O "$target_dir/$gdino_model" "$gdino_url"
fi

tap_model="tap_vit_l_v1_1.pkl"
tap_url="https://huggingface.co/BAAI/tokenize-anything/resolve/main/models/tap_vit_l_v1_1.pkl"

if [ ! -f "$target_dir/$tap_model" ]; then
    echo "Downloading $tap_model to $target_dir/$tap_model"
    wget -O "$target_dir/$tap_model" "$tap_url"
fi