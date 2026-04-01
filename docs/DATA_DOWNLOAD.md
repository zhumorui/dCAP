# Data & Weights

## Google Drive Links
- Dataset: https://drive.google.com/drive/folders/1ncVZwdH0--EWvShde8BsOGoCXWzUMf56?usp=sharing
- Checkpoints: https://drive.google.com/drive/folders/1SputNbvpapvBGawSY8JHiie_lNSdim8V?usp=sharing

## Expected Layout
```text
ckpts/
  dcap.pth
  bevformer_epoch24.pth

data/
  stt4at/
    maps/
    samples/
    v1.0-mini/
      attribute.json
      calibrated_sensor.json
      category.json
      ego_pose.json
      instance.json
      log.json
      map.json
      sample.json
      sample_annotation.json
      sample_data.json
      scene.json
      sensor.json
      visibility.json
    splits.py
  infos_mini/
    stt4at_mini_infos_temporal_train_mono3d.coco.json
    stt4at_mini_infos_temporal_train.pkl
    stt4at_mini_infos_temporal_val_mono3d.coco.json
    stt4at_mini_infos_temporal_val.pkl
```

## Notes
- The released `v1.0-mini/calibrated_sensor.json` corresponds to the ground-truth calibration annotation used in the paper release.

## Dataset Format

Camera-pose root:
- `data/stt4at/`

Camera order:
- `CAM_FRONT`
- `CAM_FRONT_LEFT`
- `CAM_FRONT_RIGHT`
- `CAM_BACK`
- `CAM_BACK_LEFT`
- `CAM_BACK_RIGHT`

BEVFormer temporal info root:
- `data/infos_mini/`

Required temporal info files:
- `stt4at_mini_infos_temporal_train.pkl`
- `stt4at_mini_infos_temporal_val.pkl`

If you need to regenerate temporal info files, use:

```bash
python dcap/perception/bevformer/create_data.py nuscenes \
  --root-path data/stt4at \
  --version v1.0-mini \
  --out-dir data/infos_mini \
  --extra-tag stt4at_mini
```
