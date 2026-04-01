#!/usr/bin/env python3
"""
Use VGGT to predict trailer camera poses and generate calibrated_sensor_vggt.json
This allows using existing BEVFormer models without any code modifications.
Process all scenes from mini_val split automatically.
"""

import os
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import sys

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# nuScenes splits (installed via pip)
from nuscenes.utils.splits import mini_val


def compute_scale_from_poses(vggt_poses: torch.Tensor, 
                            gt_poses: torch.Tensor, 
                            method: str = 'median') -> float:
    """Compute scale factor between VGGT normalized and GT metric poses."""
    vggt_translations = vggt_poses[:, :3, 3]
    gt_translations = gt_poses[:, :3, 3]
    
    vggt_dists = torch.cdist(vggt_translations, vggt_translations)
    gt_dists = torch.cdist(gt_translations, gt_translations)
    
    mask = torch.triu(torch.ones_like(vggt_dists, dtype=torch.bool), diagonal=1)
    vggt_dists_valid = vggt_dists[mask]
    gt_dists_valid = gt_dists[mask]
    
    valid_mask = (vggt_dists_valid > 1e-6) & (gt_dists_valid > 1e-6)
    if valid_mask.sum() == 0:
        return 1.0
        
    vggt_dists_valid = vggt_dists_valid[valid_mask]
    gt_dists_valid = gt_dists_valid[valid_mask]
    
    scale_ratios = gt_dists_valid / vggt_dists_valid
    
    if method == 'median':
        return torch.median(scale_ratios).item()
    elif method == 'mean':
        return torch.mean(scale_ratios).item()
    else:
        return (torch.sum(vggt_dists_valid * gt_dists_valid) / 
                torch.sum(vggt_dists_valid ** 2)).item()


def align_poses_procrustes(vggt_poses: torch.Tensor, 
                          gt_poses: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Align VGGT poses to GT poses using Procrustes analysis."""
    vggt_translations = vggt_poses[:, :3, 3]
    gt_translations = gt_poses[:, :3, 3]
    
    vggt_centroid = torch.mean(vggt_translations, dim=0)
    gt_centroid = torch.mean(gt_translations, dim=0)
    
    vggt_centered = vggt_translations - vggt_centroid
    gt_centered = gt_translations - gt_centroid
    
    vggt_scale = torch.sqrt(torch.sum(vggt_centered ** 2))
    gt_scale = torch.sqrt(torch.sum(gt_centered ** 2))
    
    if vggt_scale < 1e-8:
        return vggt_poses, 1.0
    
    scale = (gt_scale / vggt_scale).item()
    
    aligned_poses = vggt_poses.clone()
    aligned_poses[:, :3, 3] = (vggt_translations - vggt_centroid) * scale + gt_centroid
    
    return aligned_poses, scale


class VGGTTrailerCalibrator:
    """Use VGGT to calibrate trailer camera poses."""
    
    def __init__(self, 
                 data_root: str,
                 model_name: str = 'facebook/VGGT-1B',
                 device: str = 'cuda',
                 known_cameras: List[str] = None,
                 unknown_cameras: List[str] = None,
                 scale_method: str = 'procrustes'):
        
        self.data_root = Path(data_root)
        self.device = device
        self.scale_method = scale_method
        
        # Default camera mapping
        self.known_cameras = known_cameras or ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.unknown_cameras = unknown_cameras or ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.all_cameras = self.known_cameras + self.unknown_cameras
        
        # Load VGGT model
        print(f"Loading VGGT model: {model_name}")
        self.model = VGGT.from_pretrained(model_name).to(device)
        self.model.eval()
        # Determine dtype
        if torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
    
    def load_scene_data(self) -> Dict:
        """Load scene data including samples and sensor calibrations."""
        version_path = self.data_root / 'v1.0-mini'
        
        # Load metadata files
        with open(version_path / 'scene.json', 'r') as f:
            scenes = json.load(f)
        with open(version_path / 'sample.json', 'r') as f:
            samples = json.load(f)
        with open(version_path / 'sample_data.json', 'r') as f:
            sample_data = json.load(f)
        with open(version_path / 'calibrated_sensor.json', 'r') as f:
            calibrated_sensors = json.load(f)
        with open(version_path / 'ego_pose.json', 'r') as f:
            ego_poses = json.load(f)
        
        return {
            'scenes': scenes,
            'samples': samples,
            'sample_data': sample_data,
            'calibrated_sensors': calibrated_sensors,
            'ego_poses': ego_poses
        }

    def resolve_reference_calibration_path(self) -> Path:
        """Resolve the calibration file used as the update template."""
        version_path = self.data_root / 'v1.0-mini'
        candidates = [
            version_path / 'calibrated_sensor.release_backup.json',
            version_path / 'calibrated_sensor_gt.json',
            version_path / 'calibrated_sensor.json',
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"No calibration file found under {version_path}")
    
    def get_camera_images_for_sample(self, sample_token: str, scene_data: Dict) -> List[str]:
        """Get image paths for all cameras in a sample."""
        samples = {s['token']: s for s in scene_data['samples']}
        sample_data = scene_data['sample_data']
        
        sample = samples[sample_token]
        image_paths = []
        
        # Group sample_data by camera type for this sample
        sample_cameras = {}
        for sd in sample_data:
            if sd['sample_token'] == sample_token:
                filename = sd['filename']
                if 'CAM_' in filename:
                    cam_dir = filename.split('/')[1]  # samples/CAM_FRONT/... -> CAM_FRONT
                    sample_cameras[cam_dir] = sd
        
        for cam_name in self.all_cameras:
            if cam_name not in sample_cameras:
                raise ValueError(f"Camera {cam_name} not found for sample {sample_token}")
            cam_data = sample_cameras[cam_name]
            image_path = self.data_root / cam_data['filename']
            image_paths.append(str(image_path))
        
        return image_paths
    
    def get_known_camera_poses(self, sample_token: str, scene_data: Dict) -> torch.Tensor:
        """Get ground truth poses for known cameras."""
        sample_data = scene_data['sample_data']
        calibrated_sensors = {s['token']: s for s in scene_data['calibrated_sensors']}
        ego_poses = {s['token']: s for s in scene_data['ego_poses']}
        poses = []
        
        # Group sample_data by camera type for this sample
        sample_cameras = {}
        for sd in sample_data:
            if sd['sample_token'] == sample_token:
                filename = sd['filename']
                if 'CAM_' in filename:
                    cam_dir = filename.split('/')[1]  # samples/CAM_FRONT/... -> CAM_FRONT
                    sample_cameras[cam_dir] = sd
        
        for cam_name in self.known_cameras:
            if cam_name not in sample_cameras:
                raise ValueError(f"Camera {cam_name} not found for sample {sample_token}")
                
            cam_data = sample_cameras[cam_name]
            
            # Get calibrated sensor data
            calib_sensor = calibrated_sensors[cam_data['calibrated_sensor_token']]
            
            # Sensor to ego transformation
            sensor2ego_rotation = np.array(calib_sensor['rotation'])
            sensor2ego_translation = np.array(calib_sensor['translation'])
            
            # Create transformation matrix
            pose_matrix = np.eye(4)
            # Convert quaternion to rotation matrix (assuming [w,x,y,z] format)
            from scipy.spatial.transform import Rotation as R
            if len(sensor2ego_rotation) == 4:
                rot = R.from_quat([sensor2ego_rotation[1], sensor2ego_rotation[2], 
                                  sensor2ego_rotation[3], sensor2ego_rotation[0]])
                pose_matrix[:3, :3] = rot.as_matrix()
            else:
                pose_matrix[:3, :3] = np.array(sensor2ego_rotation).reshape(3, 3)
            
            pose_matrix[:3, 3] = sensor2ego_translation
            poses.append(pose_matrix[:3, :])  # [3, 4]
        
        return torch.tensor(np.stack(poses), dtype=torch.float32, device=self.device)
    
    def get_all_camera_poses(self, sample_token: str, scene_data: Dict) -> torch.Tensor:
        """Get ground truth poses for all cameras (for debugging/comparison)."""
        sample_data = scene_data['sample_data']
        calibrated_sensors = {s['token']: s for s in scene_data['calibrated_sensors']}
        ego_poses = {s['token']: s for s in scene_data['ego_poses']}
        poses = []
        
        # Group sample_data by camera type for this sample
        sample_cameras = {}
        for sd in sample_data:
            if sd['sample_token'] == sample_token:
                filename = sd['filename']
                if 'CAM_' in filename:
                    cam_dir = filename.split('/')[1]  # samples/CAM_FRONT/... -> CAM_FRONT
                    sample_cameras[cam_dir] = sd
        
        for cam_name in self.all_cameras:
            if cam_name not in sample_cameras:
                raise ValueError(f"Camera {cam_name} not found for sample {sample_token}")
                
            cam_data = sample_cameras[cam_name]
            
            # Get calibrated sensor data
            calib_sensor = calibrated_sensors[cam_data['calibrated_sensor_token']]
            
            # Sensor to ego transformation
            sensor2ego_rotation = np.array(calib_sensor['rotation'])
            sensor2ego_translation = np.array(calib_sensor['translation'])
            
            # Create transformation matrix
            pose_matrix = np.eye(4)
            # Convert quaternion to rotation matrix (assuming [w,x,y,z] format)
            from scipy.spatial.transform import Rotation as R
            if len(sensor2ego_rotation) == 4:
                rot = R.from_quat([sensor2ego_rotation[1], sensor2ego_rotation[2], 
                                  sensor2ego_rotation[3], sensor2ego_rotation[0]])
                pose_matrix[:3, :3] = rot.as_matrix()
            else:
                pose_matrix[:3, :3] = np.array(sensor2ego_rotation).reshape(3, 3)
            
            pose_matrix[:3, 3] = sensor2ego_translation
            poses.append(pose_matrix[:3, :])  # [3, 4]
        
        return torch.tensor(np.stack(poses), dtype=torch.float32, device=self.device)
    
    def predict_camera_poses(self, image_paths: List[str]) -> torch.Tensor:
        """Predict camera poses using VGGT."""
        # Load and preprocess images
        processed_images = load_and_preprocess_images(image_paths).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                # Get aggregated tokens
                aggregated_tokens_list, ps_idx = self.model.aggregator(processed_images[None])
                
                # Predict camera poses
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                
                # Convert to extrinsics and intrinsics
                image_size = (900, 1600)  # H, W
                extrinsics, intrinsics = pose_encoding_to_extri_intri(
                    pose_enc, 
                    image_size_hw=image_size,
                    pose_encoding_type="absT_quaR_FoV"
                )
        
        return extrinsics.squeeze(0)  # Remove batch dimension
    
    def align_and_predict_unknown_poses(self, vggt_poses: torch.Tensor, gt_poses: torch.Tensor) -> torch.Tensor:
        """Align VGGT poses and predict unknown camera poses."""
        known_indices = list(range(len(self.known_cameras)))
        
        known_vggt_poses = vggt_poses[known_indices]
        known_gt_poses = gt_poses
        
        if self.scale_method == 'procrustes':
            aligned_poses, scale = align_poses_procrustes(known_vggt_poses, known_gt_poses)
            # Apply same transformation to all poses
            vggt_centroid = torch.mean(known_vggt_poses[:, :3, 3], dim=0)
            gt_centroid = torch.mean(known_gt_poses[:, :3, 3], dim=0)
            aligned_all_poses = vggt_poses.clone()
            aligned_all_poses[:, :3, 3] = (vggt_poses[:, :3, 3] - vggt_centroid) * scale + gt_centroid
        else:
            scale = compute_scale_from_poses(known_vggt_poses, known_gt_poses, self.scale_method)
            aligned_all_poses = vggt_poses.clone()
            aligned_all_poses[:, :3, 3] *= scale
        
        return aligned_all_poses
    
    def find_scene_token_by_name(self, scene_name: str, scene_data: Dict) -> str:
        """Find scene token by scene name."""
        for scene in scene_data['scenes']:
            if scene['name'] == scene_name:
                return scene['token']
        raise ValueError(f"Scene {scene_name} not found")
    
    def calibrate_scene(self, scene_token: str) -> Dict:
        """Calibrate all samples in a scene."""
        print(f"Calibrating scene: {scene_token}")
        
        # Load scene data
        scene_data = self.load_scene_data()
        
        # Find scene
        scene = None
        for s in scene_data['scenes']:
            if s['token'] == scene_token:
                scene = s
                break
        
        if scene is None:
            raise ValueError(f"Scene {scene_token} not found")
        
        # Get all samples in scene
        samples = []
        current_sample = scene['first_sample_token']
        while current_sample != '':
            sample = None
            for s in scene_data['samples']:
                if s['token'] == current_sample:
                    sample = s
                    break
            if sample is None:
                break
            samples.append(sample)
            current_sample = sample['next']
        
        print(f"Found {len(samples)} samples in scene")
        
        # Process each sample
        calibrated_results = {}
        for sample in tqdm(samples, desc="Processing samples"):
            sample_token = sample['token']
            
            try:
                # Get image paths
                image_paths = self.get_camera_images_for_sample(sample_token, scene_data)
                
                # Get ground truth poses for known cameras (for alignment)
                known_gt_poses = self.get_known_camera_poses(sample_token, scene_data)
                
                # Get all camera poses (for debugging/comparison)
                all_gt_poses = self.get_all_camera_poses(sample_token, scene_data)
                
                # Predict poses with VGGT
                vggt_poses = self.predict_camera_poses(image_paths)
                
                # Align and get calibrated poses
                aligned_poses = self.align_and_predict_unknown_poses(vggt_poses, known_gt_poses)
                
                calibrated_results[sample_token] = {
                    'original_poses': all_gt_poses.cpu().numpy(),
                    'vggt_poses': vggt_poses.cpu().numpy(),
                    'calibrated_poses': aligned_poses.cpu().numpy(),
                }
                
            except Exception as e:
                print(f"Error processing sample {sample_token}: {e}")
                continue
        
        return calibrated_results
    
    def update_calibrated_sensor_json(self, all_calibrated_results: Dict):
        """Generate calibrated_sensor.json with new trailer camera poses from VGGT."""
        version_path = self.data_root / 'v1.0-mini'
        
        reference_path = self.resolve_reference_calibration_path()

        # Load existing calibrated sensors from the best available reference file.
        with open(reference_path, 'r') as f:
            calibrated_sensors = json.load(f)
        print(f"Using calibration template: {reference_path.name}")
        
        # Load scene data for mapping
        scene_data = self.load_scene_data()
        sample_data = scene_data['sample_data']
        
        # Update trailer camera calibrations for all scenes
        unknown_cam_indices = list(range(len(self.known_cameras), len(self.all_cameras)))
        updated_sensor_tokens = set()
        
        print("Updating calibrated sensor data...")
        for scene_results in all_calibrated_results.values():
            for sample_token, results in scene_results.items():
                calibrated_poses = results['calibrated_poses']
                
                # Group sample_data by camera type for this sample
                sample_cameras = {}
                for sd in sample_data:
                    if sd['sample_token'] == sample_token:
                        filename = sd['filename']
                        if 'CAM_' in filename:
                            cam_dir = filename.split('/')[1]  # samples/CAM_FRONT/... -> CAM_FRONT
                            sample_cameras[cam_dir] = sd
                
                for i, cam_idx in enumerate(unknown_cam_indices):
                    cam_name = self.unknown_cameras[i]
                    calibrated_pose = calibrated_poses[cam_idx]  # [3, 4]
                    
                    if cam_name not in sample_cameras:
                        continue  # Skip if camera not found for this sample
                    
                    # Get sample data for this camera
                    cam_data = sample_cameras[cam_name]
                    sensor_token = cam_data['calibrated_sensor_token']
                    
                    if sensor_token in updated_sensor_tokens:
                        continue  # Already updated this sensor
                    
                    # Convert pose matrix back to quaternion + translation format
                    from scipy.spatial.transform import Rotation as R
                    rotation_matrix = calibrated_pose[:3, :3]
                    translation = calibrated_pose[:3, 3].tolist()
                    
                    rot = R.from_matrix(rotation_matrix)
                    quaternion = rot.as_quat()  # [x, y, z, w]
                    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # [w, x, y, z]
                    
                    # Find and update corresponding calibrated sensor
                    for sensor in calibrated_sensors:
                        if sensor['token'] == sensor_token:
                            sensor['rotation'] = quaternion
                            sensor['translation'] = translation
                            updated_sensor_tokens.add(sensor_token)
                            break
        
        print(f"Updated {len(updated_sensor_tokens)} trailer camera sensors")
        
        # Save updated calibrated sensors to calibrated_sensor.json for visualization
        output_path = version_path / 'calibrated_sensor.json'
        with open(output_path, 'w') as f:
            json.dump(calibrated_sensors, f, indent=2)
        
        print(f"Generated calibrated_sensor.json: {output_path}")
        
    def calibrate_all_mini_val_scenes(self) -> Dict:
        """Calibrate all scenes in mini_val split."""
        print(f"Starting VGGT calibration for {len(mini_val)} scenes from mini_val split")
        
        # Load scene data once
        scene_data = self.load_scene_data()
        
        all_results = {}
        for scene_name in mini_val:
            try:
                print(f"\n=== Processing scene: {scene_name} ===")
                scene_token = self.find_scene_token_by_name(scene_name, scene_data)
                scene_results = self.calibrate_scene(scene_token)
                all_results[scene_name] = scene_results
            except Exception as e:
                print(f"Error processing scene {scene_name}: {e}")
                continue
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Calibrate trailer cameras using VGGT for all mini_val scenes')
    parser.add_argument('--data-root', type=str, required=True, 
                       help='Root path to the STT4AT dataset')
    parser.add_argument('--model-name', type=str, default='facebook/VGGT-1B',
                       help='VGGT model name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run VGGT on')
    parser.add_argument('--scale-method', type=str, default='procrustes',
                       choices=['procrustes', 'median', 'mean'],
                       help='Scale alignment method')
    
    args = parser.parse_args()
    
    print("=== VGGT Trailer Camera Calibration ===")
    print(f"Data root: {args.data_root}")
    print(f"Processing scenes: {mini_val}")
    print(f"Scale method: {args.scale_method}")
    
    calibrator = VGGTTrailerCalibrator(
        data_root=args.data_root,
        model_name=args.model_name,
        device=args.device,
        scale_method=args.scale_method
    )
    
    # Calibrate all mini_val scenes
    all_calibrated_results = calibrator.calibrate_all_mini_val_scenes()
    
    # Generate calibrated_sensor_vggt.json
    calibrator.update_calibrated_sensor_json(all_calibrated_results)
    
    print("\n=== Calibration Summary ===")
    total_scenes = len(all_calibrated_results)
    total_samples = sum(len(scene_results) for scene_results in all_calibrated_results.values())
    
    print(f"Processed {total_scenes}/{len(mini_val)} scenes successfully")
    print(f"Total samples calibrated: {total_samples}")
    print(f"Generated: {args.data_root}/v1.0-mini/calibrated_sensor.json")
    print("\nYou can now use existing BEVFormer configs directly.")
    print("The trailer camera poses have been updated with VGGT predictions.")


if __name__ == '__main__':
    main()
