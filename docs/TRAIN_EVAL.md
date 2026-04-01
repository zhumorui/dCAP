# Train & Eval

This document covers the full release workflow:

1. Train dCAP
2. Evaluate dCAP
3. Export a predicted `calibrated_sensor.json`
4. Replace the calibration file and regenerate temporal infos
5. Evaluate BEVFormer with the updated calibration

## Prerequisites

- Follow [INSTALL.md](INSTALL.md)
- Download the dataset and checkpoints from [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)
- Arrange files exactly as described in the expected layout

## 1. Train dCAP

Use the released camera-pose config:

```bash
python -m dcap.camera_pose.train \
  --config configs/camera_pose/truck_trailer.yaml
```

By default this writes training outputs to:

```text
work_dirs/camera_pose_v3_jan_24/
```

## 2. Evaluate dCAP

Evaluate a trained or released checkpoint with:

```bash
python -m dcap.camera_pose.eval \
  --config configs/camera_pose/truck_trailer.yaml \
  --checkpoint ckpts/dcap.pth
```

This step reports camera-pose metrics only. It does not modify dataset files and it does not export a new calibration JSON.

## 3. Export Predicted Calibration

Generate a predicted trailer-camera calibration file with:

```bash
python -m dcap.camera_pose.scripts.export_predicted_calibration \
  --config configs/camera_pose/truck_trailer.yaml \
  --checkpoint ckpts/dcap.pth \
  --output outputs/calibration/calibrated_sensor.json
```

The exported file is a full `calibrated_sensor.json` copy with updated trailer-camera entries:

- `CAM_BACK`
- `CAM_BACK_LEFT`
- `CAM_BACK_RIGHT`

The script predicts `CAM_BACK` and propagates the fixed trailer-camera relative transforms to `CAM_BACK_LEFT` and `CAM_BACK_RIGHT`.

## 4. Replace Calibration and Regenerate Temporal Infos

BEVFormer does not read camera calibration directly from `calibrated_sensor.json` during evaluation. It reads temporal info files that were generated from the dataset metadata. This means the new calibration will not take effect until those temporal infos are regenerated.

Back up the released calibration file first:

```bash
cp data/stt4at/v1.0-mini/calibrated_sensor.json \
  data/stt4at/v1.0-mini/calibrated_sensor.release_backup.json
```

Replace it with the exported prediction:

```bash
cp outputs/calibration/calibrated_sensor.json \
  data/stt4at/v1.0-mini/calibrated_sensor.json
```

Remove the existing temporal info files:

```bash
rm -f data/infos_mini/stt4at_mini_infos_temporal_train.pkl
rm -f data/infos_mini/stt4at_mini_infos_temporal_val.pkl
rm -f data/infos_mini/stt4at_mini_infos_temporal_train_mono3d.coco.json
rm -f data/infos_mini/stt4at_mini_infos_temporal_val_mono3d.coco.json
```

Regenerate them with:

```bash
python dcap/perception/bevformer/create_data.py nuscenes \
  --root-path data/stt4at \
  --version v1.0-mini \
  --out-dir data/infos_mini \
  --extra-tag stt4at_mini
```

This refreshes:

- `stt4at_mini_infos_temporal_train.pkl`
- `stt4at_mini_infos_temporal_val.pkl`
- `stt4at_mini_infos_temporal_train_mono3d.coco.json`
- `stt4at_mini_infos_temporal_val_mono3d.coco.json`

## 5. Evaluate BEVFormer with Updated Calibration

Once `calibrated_sensor.json` has been replaced and the temporal infos have been regenerated, run BEVFormer evaluation with:

```bash
bash dcap/perception/bevformer/dist_test.sh \
  configs/perception/bevformer_base.py \
  ckpts/bevformer_epoch24.pth \
  8
```

If you skip the regeneration step, BEVFormer evaluation will still use temporal infos derived from the old calibration.

## Optional: Train BEVFormer with Updated Calibration

If you want to train BEVFormer using the updated calibration instead of only evaluating it, regenerate the temporal infos first, then launch training:

```bash
bash dcap/perception/bevformer/dist_train.sh \
  configs/perception/bevformer_base.py \
  8
```

## Restore the Released Calibration

To switch back to the released calibration:

```bash
cp data/stt4at/v1.0-mini/calibrated_sensor.release_backup.json \
  data/stt4at/v1.0-mini/calibrated_sensor.json
```

Then regenerate the temporal infos again with the same `create_data.py` command.
