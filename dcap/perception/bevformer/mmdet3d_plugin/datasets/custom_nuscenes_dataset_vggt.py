import numpy as np
import torch
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmcv.datasets.custom_nuscenes_dataset import CustomNuScenesDataset


@DATASETS.register_module()
class CustomNuScenesDatasetVGGT(CustomNuScenesDataset):
    """
    Extended CustomNuScenesDataset with VGGT camera pose prediction integration.
    Handles dynamic camera poses for trailer cameras using VGGT predictions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_vggt_poses = True
    
    def get_data_info(self, index):
        """
        Get data info with support for VGGT predicted camera poses.
        """
        info = self.data_infos[index]
        
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )
        
        # Add camera information with support for VGGT poses
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT', 
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
        
        img_filenames = []
        lidar2img = []
        cam2img = []
        cam_intrinsics = []
        cam_extrinsics = []
        
        for cam_type in camera_types:
            cam_info = info['cams'][cam_type]
            
            img_filenames.append(cam_info['data_path'])
            
            # Camera intrinsics
            cam_intrinsic = np.array(cam_info['cam_intrinsic'])
            cam_intrinsics.append(cam_intrinsic)
            
            # Camera extrinsics (sensor2ego transformation)
            cam_extrinsic = np.array(cam_info['sensor2ego_rotation']).T
            cam_translation = np.array(cam_info['sensor2ego_translation'])
            cam_extrinsic_matrix = np.eye(4)
            cam_extrinsic_matrix[:3, :3] = cam_extrinsic
            cam_extrinsic_matrix[:3, 3] = cam_translation
            cam_extrinsics.append(cam_extrinsic_matrix[:3, :])
            
            # Lidar to image transformation
            lidar2cam = np.array(cam_info['sensor2lidar_rotation']).T
            lidar2cam_translation = np.array(cam_info['sensor2lidar_translation'])
            lidar2cam_matrix = np.eye(4)
            lidar2cam_matrix[:3, :3] = lidar2cam
            lidar2cam_matrix[:3, 3] = lidar2cam_translation
            
            # Full transformation chain
            viewpad = np.eye(4)
            viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
            lidar2img_matrix = (viewpad @ lidar2cam_matrix)
            lidar2img.append(lidar2img_matrix)
            
            # Camera to image (K matrix)
            cam2img.append(cam_intrinsic)
        
        input_dict.update(
            dict(
                img_filename=img_filenames,
                lidar2img=lidar2img,
                cam2img=cam2img,
                cam_intrinsics=cam_intrinsics,
                cam_extrinsics=cam_extrinsics,
            )
        )
        
        # Add individual camera extrinsics for VGGT processing
        for i, cam_type in enumerate(camera_types):
            input_dict[f'cam{i}_extrinsic'] = cam_extrinsics[i]
            input_dict[f'cam{i}_intrinsic'] = cam_intrinsics[i]
            input_dict[f'cam{i}_type'] = cam_type
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            
        return input_dict
    
    def __getitem__(self, idx):
        """
        Get item with VGGT pose integration.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
                
            # Process VGGT predictions if available
            if 'vggt_camera_poses' in data:
                data = self._integrate_vggt_poses(data)
                
            return data
    
    def _integrate_vggt_poses(self, data):
        """
        Integrate VGGT predicted camera poses into the data.
        Updates camera transformation matrices with predicted poses.
        """
        vggt_poses = data.get('vggt_camera_poses', {})
        predicted_poses = vggt_poses.get('predicted_poses', {})
        
        if not predicted_poses:
            return data
        
        # Update camera extrinsics with VGGT predictions
        for cam_key, predicted_pose in predicted_poses.items():
            # Extract camera index from key (e.g., 'cam3_extrinsic_predicted' -> 3)
            cam_idx = int(cam_key.split('cam')[1].split('_')[0])
            
            # Update camera extrinsic matrix
            if 'cam_extrinsics' in data:
                if isinstance(data['cam_extrinsics'], list) and len(data['cam_extrinsics']) > cam_idx:
                    data['cam_extrinsics'][cam_idx] = predicted_pose
            
            # Update lidar2img transformation
            if 'lidar2img' in data and 'cam2img' in data:
                if (isinstance(data['lidar2img'], list) and len(data['lidar2img']) > cam_idx and
                    isinstance(data['cam2img'], list) and len(data['cam2img']) > cam_idx):
                    
                    # Reconstruct lidar2img with new camera pose
                    cam_intrinsic = data['cam2img'][cam_idx]
                    
                    # Convert cam_extrinsic to lidar2cam
                    # cam_extrinsic is sensor2ego, we need sensor2lidar
                    # Assuming ego2lidar is identity or can be derived
                    sensor2ego = np.eye(4)
                    sensor2ego[:3, :] = predicted_pose
                    
                    # For simplicity, assume ego frame aligns with lidar frame
                    sensor2lidar = sensor2ego  # This might need adjustment based on your setup
                    
                    # Invert to get lidar2sensor
                    lidar2cam_matrix = np.linalg.inv(sensor2lidar)
                    
                    # Update lidar2img
                    viewpad = np.eye(4)
                    viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
                    data['lidar2img'][cam_idx] = viewpad @ lidar2cam_matrix
        
        return data
    
    def evaluate_vggt_poses(self, results, metric='pose_error', **kwargs):
        """
        Evaluate VGGT camera pose predictions.
        
        Args:
            results: List of prediction results
            metric: Evaluation metric ('pose_error', 'translation_error', 'rotation_error')
        
        Returns:
            dict: Evaluation results
        """
        evaluation_results = {}
        
        if metric == 'pose_error' or 'pose' in metric:
            translation_errors = []
            rotation_errors = []
            
            for result in results:
                if 'vggt_camera_poses' in result:
                    vggt_poses = result['vggt_camera_poses']
                    predicted_poses = vggt_poses.get('predicted_poses', {})
                    
                    # Compare with ground truth if available
                    for cam_key, predicted_pose in predicted_poses.items():
                        cam_idx = int(cam_key.split('cam')[1].split('_')[0])
                        
                        # Get ground truth pose (if available in test data)
                        gt_key = f'cam{cam_idx}_extrinsic'
                        if gt_key in result:
                            gt_pose = result[gt_key]
                            
                            # Compute translation error
                            trans_error = np.linalg.norm(predicted_pose[:3, 3] - gt_pose[:3, 3])
                            translation_errors.append(trans_error)
                            
                            # Compute rotation error (Frobenius norm of rotation difference)
                            rot_error = np.linalg.norm(predicted_pose[:3, :3] - gt_pose[:3, :3], 'fro')
                            rotation_errors.append(rot_error)
            
            if translation_errors:
                evaluation_results.update({
                    'translation_error_mean': np.mean(translation_errors),
                    'translation_error_std': np.std(translation_errors),
                    'translation_error_median': np.median(translation_errors),
                    'rotation_error_mean': np.mean(rotation_errors),
                    'rotation_error_std': np.std(rotation_errors),
                    'rotation_error_median': np.median(rotation_errors),
                })
        
        return evaluation_results