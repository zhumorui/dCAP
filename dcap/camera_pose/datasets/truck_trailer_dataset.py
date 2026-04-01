# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from dcap.camera_pose.datasets.base_dataset import BaseDataset
from dcap.camera_pose.datasets.dataset_util import *  # noqa: F401,F403 - re-exported helpers
from dcap.camera_pose.utils.pose_enc import extri_intri_to_pose_encoding





def _get_conf_value(cfg, name, default=None):
    """Return config attr supporting both dict and SimpleNamespace."""
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


class TruckTrailerDataset(BaseDataset):
    """
    Dataset for truck-trailer with articulated camera poses.
    
    Handles 6 cameras:
    - Front 3 cameras (truck): fixed poses relative to ego
    - Back 3 cameras (trailer): variable poses due to articulation  
    
    Data format follows nuScenes structure with additional trailer dynamics.
    """
    
    def __init__(self, common_conf, dataset_conf):
        super().__init__(common_conf)
        
        self.data_root = _get_conf_value(dataset_conf, 'data_root')
        if self.data_root is None:
            raise ValueError('dataset_conf must provide data_root')
        self.version = _get_conf_value(dataset_conf, 'version', 'v1.0-mini')
        self.seq_len = _get_conf_value(dataset_conf, 'seq_len', 6)  # Number of cameras
        self.sample_stride = _get_conf_value(dataset_conf, 'sample_stride', 1)
        self.split = _get_conf_value(dataset_conf, 'split', 'train')
        self.queue_length = int(_get_conf_value(dataset_conf, 'queue_length', 1))
        if self.queue_length < 1:
            raise ValueError('queue_length must be >= 1')

        # Load nuScenes-style metadata
        self.version_path = os.path.join(self.data_root, self.version)
        self._load_metadata()

        # Camera ordering (important for truck vs trailer distinction)
        self.camera_order = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',    # Truck cameras (0-2)
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'        # Trailer cameras (3-5)
        ]
        
        # Load scene splits if available
        self.allowed_scenes = self._load_scene_splits()

        # Initialize samples for the requested split
        self.samples = self._prepare_split_samples()
        self.len_train = len(self.samples)
        if self.len_train == 0:
            raise ValueError(f"No samples found for split '{self.split}' in {self.version_path}")

    def _load_metadata(self):
        """Load nuScenes-style JSON metadata files."""
        metadata_files = [
            'sample.json', 'sample_data.json', 'calibrated_sensor.json',
            'ego_pose.json', 'scene.json'
        ]
        
        self.metadata = {}
        for file_name in metadata_files:
            file_path = os.path.join(self.version_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.metadata[file_name.replace('.json', '')] = json.load(f)
                    
        # Create lookup dictionaries for fast access
        self.sample_dict = {item['token']: item for item in self.metadata['sample']}
        self.sample_data_dict = {item['token']: item for item in self.metadata['sample_data']}
        self.calib_sensor_dict = {item['token']: item for item in self.metadata['calibrated_sensor']}
        self.ego_pose_dict = {item['token']: item for item in self.metadata['ego_pose']}
        self.scene_dict = {item['token']: item for item in self.metadata['scene']}

    def _load_scene_splits(self):
        """Load scene name splits from data_root/splits.py if present."""
        splits_path = Path(self.data_root) / 'splits.py'
        if not splits_path.exists():
            return None

        spec = importlib.util.spec_from_file_location('truck_trailer_splits', splits_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        split_map = {
            'train': getattr(module, 'TRAIN_SCENES', None),
            'val': getattr(module, 'VAL_SCENES', None),
            'test': getattr(module, 'TEST_SCENES', None) if hasattr(module, 'TEST_SCENES') else None,
        }

        scenes = split_map.get(self.split)
        if scenes:
            return set(scenes)
        return None
        
    def _prepare_split_samples(self):
        """Prepare list of valid samples for the configured split."""
        samples = []

        for sample in self.metadata['sample']:
            sample_token = sample['token']

            if self.allowed_scenes is not None:
                scene_token = sample.get('scene_token')
                scene_name = None
                if scene_token:
                    scene = self.scene_dict.get(scene_token)
                    if scene:
                        scene_name = scene.get('name')
                if scene_name is None or scene_name not in self.allowed_scenes:
                    continue
            
            # Collect camera data for this sample
            camera_data = {}
            for sd in self.metadata['sample_data']:
                if sd['sample_token'] != sample_token:
                    continue
                filename = sd.get('filename', '')
                parts = filename.split('/') if filename else []
                camera_name = parts[1] if len(parts) > 1 else None
                if camera_name in self.camera_order:
                    camera_data[camera_name] = sd
            
            # Only include samples with all 6 cameras
            if len(camera_data) == len(self.camera_order):
                samples.append({
                    'sample_token': sample_token,
                    'camera_data': camera_data,
                    'scene_token': sample.get('scene_token'),
                })

        print(f"Loaded {len(samples)} {self.split} samples from truck-trailer dataset")
        return samples
        
    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0, img_per_seq=None, **kwargs):
        """
        Get training data for a single sample.
        
        Returns:
            dict: Contains images, poses, intrinsics, and other training data
        """
        if seq_index is None:
            seq_index = 0
            
        sample_info = self.samples[seq_index]
        camera_data = sample_info['camera_data']
        scene_token = sample_info.get('scene_token')

        history_indices, history_mask = self._gather_history_indices(seq_index, scene_token)

        frame_tensors = [self._process_frame(self.samples[idx]['camera_data'], aspect_ratio) for idx in history_indices]

        images = torch.stack([frame['images'] for frame in frame_tensors], dim=0)  # [T, 6, 3, H, W]
        extrinsics = torch.stack([frame['extrinsics'] for frame in frame_tensors], dim=0)  # [T, 6, 3, 4]
        intrinsics = torch.stack([frame['intrinsics'] for frame in frame_tensors], dim=0)  # [T, 6, 3, 3]
        depths = torch.stack([frame['depths'] for frame in frame_tensors], dim=0)  # [T, 6, H, W]
        valid_masks = torch.stack([frame['point_masks'] for frame in frame_tensors], dim=0)  # [T, 6, H, W]
        ego_poses = torch.stack([frame['ego_pose'] for frame in frame_tensors], dim=0)  # [T, 4]

        frame_mask = torch.tensor(history_mask, dtype=torch.bool)

        fixed_truck_poses, fixed_trailer_poses = self._get_fixed_poses(intrinsics[-1], images.shape[-2:])

        return {
            'images': images,           # [T, 6, 3, H, W]
            'extrinsics': extrinsics,   # [T, 6, 3, 4]
            'intrinsics': intrinsics,   # [T, 6, 3, 3]
            'depths': depths,           # [T, 6, H, W]
            'point_masks': valid_masks, # [T, 6, H, W]
            'ego_poses': ego_poses,     # [T, 4]
            'frame_mask': frame_mask,   # [T]
            'fixed_truck_poses': fixed_truck_poses,    # [3, 9]
            'fixed_trailer_poses': fixed_trailer_poses, # [3, 9]
            'seq_name': sample_info['sample_token'],
            'camera_names': self.camera_order,
            'cam_ids': torch.arange(len(self.camera_order), dtype=torch.long),
        }

    def _gather_history_indices(self, seq_index, scene_token):
        history_indices = []
        history_mask = []
        current_idx = seq_index
        while len(history_indices) < self.queue_length:
            history_indices.append(current_idx)
            history_mask.append(True)
            if len(history_indices) == self.queue_length:
                break
            prev_idx = current_idx - 1
            if prev_idx < 0:
                break
            prev_scene = self.samples[prev_idx].get('scene_token')
            if scene_token is not None and prev_scene != scene_token:
                break
            current_idx = prev_idx

        if len(history_indices) < self.queue_length:
            first_valid = history_indices[-1]
            while len(history_indices) < self.queue_length:
                history_indices.append(first_valid)
                history_mask.append(False)

        history_indices.reverse()
        history_mask.reverse()
        return history_indices, history_mask

    def _process_frame(self, camera_data, aspect_ratio):
        images = []
        extrinsics = []
        intrinsics = []
        depths = []
        valid_masks = []
        ego_pose_vec = None

        for i, camera_name in enumerate(self.camera_order):
            if camera_name not in camera_data:
                raise ValueError(f"Missing camera data for {camera_name}")
                
            sample_data = camera_data[camera_name]
            
            # Load image
            filename = sample_data.get('filename')
            if filename is None:
                raise KeyError(f'Sample data for {camera_name} missing filename')
            image_path = os.path.join(self.data_root, filename)

            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Get camera calibration
            calib_token = sample_data['calibrated_sensor_token']
            calib_data = self.calib_sensor_dict[calib_token]
            
            # Extract pose and intrinsics
            translation = np.array(calib_data['translation'])
            rotation = np.array(calib_data['rotation'])  # quaternion [w,x,y,z]
            camera_intrinsic = np.array(calib_data['camera_intrinsic'])
            
            # Convert quaternion to rotation matrix
            rotation_matrix = quat_to_rotation_matrix(rotation)
            
            # Build extrinsic matrix [R|t]
            extrinsic = np.hstack([rotation_matrix, translation.reshape(3, 1)])
            
            # Create dummy depth (since we don't have ground truth depth)
            depth = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
            
            # Get target image shape
            target_shape = self.get_target_shape(aspect_ratio)
            
            # Process this camera's data
            processed = self.process_one_image(
                image=image,
                depth_map=depth,
                extri_opencv=extrinsic,
                intri_opencv=camera_intrinsic,
                original_size=np.array(image.shape[:2]),
                target_image_shape=target_shape,
                filepath=image_path
            )
            
            (proc_image, proc_depth, proc_extrinsic, proc_intrinsic, 
             world_coords, cam_coords, point_mask, _) = processed
            
            images.append(proc_image)
            extrinsics.append(proc_extrinsic)
            intrinsics.append(proc_intrinsic)
            depths.append(proc_depth)
            valid_masks.append(point_mask)

            if ego_pose_vec is None:
                ego_token = sample_data.get('ego_pose_token')
                if ego_token is not None and ego_token in self.ego_pose_dict:
                    ego_data = self.ego_pose_dict[ego_token]
                    ego_translation = np.array(ego_data['translation'])
                    ego_rotation = np.array(ego_data['rotation'])
                    ego_rot_mat = quat_to_rotation_matrix(ego_rotation)
                    ego_yaw = np.arctan2(ego_rot_mat[1, 0], ego_rot_mat[0, 0])
                    ego_pose_vec = torch.from_numpy(
                        np.concatenate([ego_translation, [ego_yaw]], axis=0)
                    ).float()
                else:
                    ego_pose_vec = torch.zeros(4, dtype=torch.float32)
            
        # Stack into tensors
        images = np.stack(images)
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)
        depths = np.stack(depths)
        valid_masks = np.stack(valid_masks)

        images = torch.from_numpy(images).float() / 255.0
        images = images.permute(0, 3, 1, 2)
        extrinsics = torch.from_numpy(extrinsics).float()
        intrinsics = torch.from_numpy(intrinsics).float()
        depths = torch.from_numpy(depths).float()
        valid_masks = torch.from_numpy(valid_masks).bool()

        if ego_pose_vec is None:
            ego_pose_vec = torch.zeros(4, dtype=torch.float32)

        return {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'depths': depths,
            'point_masks': valid_masks,
            'ego_pose': ego_pose_vec,
        }
        
    def _get_fixed_poses(self, intrinsics, image_hw):
        """
        Get fixed camera poses for truck and trailer from dataset calibration metadata.
        
        Args:
            intrinsics: [6, 3, 3] camera intrinsics
            image_hw: (H, W) image dimensions
            
        Returns:
            tuple: (truck_poses, trailer_poses) as pose encodings
        """
        # Use the first sample as the reference for fixed poses.
        first_sample = self.samples[0]
        camera_data = first_sample['camera_data']
        
        truck_extrinsics = []
        trailer_extrinsics = []
        truck_intrinsics = []
        trailer_intrinsics = []
        
        for i, camera_name in enumerate(self.camera_order):
            sample_data = camera_data[camera_name]
            calib_token = sample_data['calibrated_sensor_token']
            
            calib_data = self.calib_sensor_dict[calib_token]
            
            translation = np.array(calib_data['translation'])
            rotation = np.array(calib_data['rotation'])
            camera_intrinsic = np.array(calib_data['camera_intrinsic'])
            
            rotation_matrix = quat_to_rotation_matrix(rotation)
            extrinsic = np.hstack([rotation_matrix, translation.reshape(3, 1)])
            
            if i < 3:  # Truck cameras
                truck_extrinsics.append(extrinsic)
                truck_intrinsics.append(camera_intrinsic)
            else:  # Trailer cameras  
                trailer_extrinsics.append(extrinsic)
                trailer_intrinsics.append(camera_intrinsic)
                
        # Convert to tensors
        truck_extrinsics = torch.from_numpy(np.stack(truck_extrinsics)).float()
        trailer_extrinsics = torch.from_numpy(np.stack(trailer_extrinsics)).float()
        truck_intrinsics = torch.from_numpy(np.stack(truck_intrinsics)).float()
        trailer_intrinsics = torch.from_numpy(np.stack(trailer_intrinsics)).float()
        
        # Convert to pose encodings
        truck_poses = extri_intri_to_pose_encoding(
            truck_extrinsics.unsqueeze(0), truck_intrinsics.unsqueeze(0), image_hw
        ).squeeze(0)  # [3, 9]
        
        trailer_poses = extri_intri_to_pose_encoding(
            trailer_extrinsics.unsqueeze(0), trailer_intrinsics.unsqueeze(0), image_hw
        ).squeeze(0)  # [3, 9]
        
        return truck_poses, trailer_poses


def quat_to_rotation_matrix(quat):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    
    Args:
        quat: quaternion as [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R
