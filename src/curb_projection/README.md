# Setup: Download weights
## MASA-GDINO
```
cd /workspaces/curb-osg-deploy-test/src/curb_projection/saved_models/masa_models
wget https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth
```

## SAM
```
cd /workspaces/curb-osg-deploy-test/src/curb_projection/saved_models/pretrain_weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

# Grounding DINO Landmark Perception
```
cd /workspaces/curb-osg-deploy-test/src/curb_projection/saved_models
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/BAAI/tokenize-anything/resolve/main/models/tap_vit_l_v1_1.pkl
```