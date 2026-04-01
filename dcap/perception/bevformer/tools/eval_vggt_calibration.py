#!/usr/bin/env python3
"""
Evaluate VGGT calibration results against ground truth for trailer cameras.
Only compares the 3 trailer cameras: CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
import pandas as pd


class VGGTCalibrationEvaluator:
    """Evaluate VGGT calibration against ground truth."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.version_path = self.data_root / 'v1.0-mini'
        
        # Trailer cameras to evaluate
        self.trailer_cameras = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
    def load_calibrated_sensors(self, filename: str) -> Dict:
        """Load calibrated sensor data from JSON file."""
        filepath = self.version_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            sensors = json.load(f)
        
        # Create mapping from sensor_token to sensor data
        sensor_dict = {sensor['token']: sensor for sensor in sensors}
        return sensor_dict

    def resolve_ground_truth_filename(self) -> str:
        """Resolve the best available ground-truth calibration file for comparison."""
        candidates = [
            'calibrated_sensor.release_backup.json',
            'calibrated_sensor_gt.json',
            'calibrated_sensor.json',
        ]
        for filename in candidates:
            if (self.version_path / filename).exists():
                return filename
        raise FileNotFoundError(f"No calibration file found under {self.version_path}")
    
    def load_sample_data(self) -> Dict:
        """Load sample data to map cameras to sensor tokens."""
        with open(self.version_path / 'sample_data.json', 'r') as f:
            sample_data = json.load(f)
        return sample_data
    
    def get_camera_sensor_tokens(self, sample_data: List[Dict]) -> Dict[str, List[str]]:
        """Get sensor tokens for each trailer camera."""
        camera_tokens = {cam: [] for cam in self.trailer_cameras}
        
        for sd in sample_data:
            filename = sd['filename']
            if 'CAM_' in filename:
                cam_name = filename.split('/')[1]  # samples/CAM_BACK/... -> CAM_BACK
                if cam_name in self.trailer_cameras:
                    token = sd['calibrated_sensor_token']
                    if token not in camera_tokens[cam_name]:
                        camera_tokens[cam_name].append(token)
        
        return camera_tokens
    
    def pose_to_matrix(self, rotation: List[float], translation: List[float]) -> np.ndarray:
        """Convert quaternion + translation to 4x4 transformation matrix."""
        matrix = np.eye(4)
        
        # Convert quaternion [w,x,y,z] to rotation matrix
        quat = np.array(rotation)
        if len(quat) == 4:
            # nuScenes format: [w,x,y,z]
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy expects [x,y,z,w]
            matrix[:3, :3] = rot.as_matrix()
        
        matrix[:3, 3] = translation
        return matrix
    
    def compute_pose_error(self, pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> Dict[str, float]:
        """Compute pose error metrics between predicted and ground truth poses."""
        # Translation error
        trans_error = np.linalg.norm(pred_matrix[:3, 3] - gt_matrix[:3, 3])
        trans_error_xyz = np.abs(pred_matrix[:3, 3] - gt_matrix[:3, 3])
        
        # Rotation error
        pred_rot = R.from_matrix(pred_matrix[:3, :3])
        gt_rot = R.from_matrix(gt_matrix[:3, :3])
        
        # Relative rotation
        rel_rot = pred_rot * gt_rot.inv()
        rot_error_deg = np.abs(rel_rot.as_rotvec())
        rot_error_angle = np.linalg.norm(rot_error_deg) * 180 / np.pi
        
        return {
            'translation_error_m': trans_error,
            'translation_error_x_m': trans_error_xyz[0],
            'translation_error_y_m': trans_error_xyz[1], 
            'translation_error_z_m': trans_error_xyz[2],
            'rotation_error_deg': rot_error_angle,
            'rotation_error_x_deg': rot_error_deg[0] * 180 / np.pi,
            'rotation_error_y_deg': rot_error_deg[1] * 180 / np.pi,
            'rotation_error_z_deg': rot_error_deg[2] * 180 / np.pi,
        }
    
    def evaluate_calibration(self) -> Dict:
        """Evaluate VGGT calibration against ground truth."""
        print("Loading calibration data...")
        
        # Load VGGT and ground truth calibrations
        vggt_sensors = self.load_calibrated_sensors('calibrated_sensor.json')
        gt_filename = self.resolve_ground_truth_filename()
        gt_sensors = self.load_calibrated_sensors(gt_filename)
        print(f"Using ground-truth calibration file: {gt_filename}")
        
        # Load sample data for camera-token mapping
        sample_data = self.load_sample_data()
        camera_tokens = self.get_camera_sensor_tokens(sample_data)
        
        print(f"Found sensor tokens for cameras:")
        for cam, tokens in camera_tokens.items():
            print(f"  {cam}: {len(tokens)} unique tokens")
        
        # Collect all errors
        all_errors = []
        camera_errors = {cam: [] for cam in self.trailer_cameras}
        
        for cam_name in self.trailer_cameras:
            print(f"\nEvaluating {cam_name}...")
            
            for token in camera_tokens[cam_name]:
                if token not in vggt_sensors:
                    print(f"  Warning: Token {token} not found in VGGT data")
                    continue
                if token not in gt_sensors:
                    print(f"  Warning: Token {token} not found in GT data")
                    continue
                
                # Get poses
                vggt_sensor = vggt_sensors[token]
                gt_sensor = gt_sensors[token]
                
                # Convert to matrices
                vggt_matrix = self.pose_to_matrix(vggt_sensor['rotation'], vggt_sensor['translation'])
                gt_matrix = self.pose_to_matrix(gt_sensor['rotation'], gt_sensor['translation'])
                
                # Compute errors
                errors = self.compute_pose_error(vggt_matrix, gt_matrix)
                errors['camera'] = cam_name
                errors['token'] = token
                
                all_errors.append(errors)
                camera_errors[cam_name].append(errors)
        
        print(f"\nTotal comparisons: {len(all_errors)}")
        return {'all_errors': all_errors, 'camera_errors': camera_errors}
    
    def compute_statistics(self, errors: List[Dict]) -> Dict:
        """Compute statistical metrics for errors."""
        if not errors:
            return {}
        
        metrics = {}
        error_keys = ['translation_error_m', 'translation_error_x_m', 'translation_error_y_m', 
                     'translation_error_z_m', 'rotation_error_deg', 'rotation_error_x_deg',
                     'rotation_error_y_deg', 'rotation_error_z_deg']
        
        for key in error_keys:
            values = [e[key] for e in errors]
            metrics[f'{key}_mean'] = np.mean(values)
            metrics[f'{key}_std'] = np.std(values)
            metrics[f'{key}_median'] = np.median(values)
            metrics[f'{key}_min'] = np.min(values)
            metrics[f'{key}_max'] = np.max(values)
            metrics[f'{key}_rmse'] = np.sqrt(np.mean(np.array(values)**2))
        
        return metrics
    
    def print_results(self, results: Dict):
        """Print evaluation results in a formatted way."""
        all_errors = results['all_errors']
        camera_errors = results['camera_errors']
        
        print("\n" + "="*80)
        print("VGGT CALIBRATION EVALUATION RESULTS")
        print("="*80)
        
        # Overall statistics
        overall_stats = self.compute_statistics(all_errors)
        print(f"\nOVERALL STATISTICS (All {len(all_errors)} trailer camera poses):")
        print("-" * 60)
        print(f"Translation Error (m):")
        print(f"  Mean: {overall_stats['translation_error_m_mean']:.4f} ± {overall_stats['translation_error_m_std']:.4f}")
        print(f"  Median: {overall_stats['translation_error_m_median']:.4f}")
        print(f"  RMSE: {overall_stats['translation_error_m_rmse']:.4f}")
        print(f"  Range: [{overall_stats['translation_error_m_min']:.4f}, {overall_stats['translation_error_m_max']:.4f}]")
        
        print(f"\nRotation Error (deg):")
        print(f"  Mean: {overall_stats['rotation_error_deg_mean']:.2f} ± {overall_stats['rotation_error_deg_std']:.2f}")
        print(f"  Median: {overall_stats['rotation_error_deg_median']:.2f}")
        print(f"  RMSE: {overall_stats['rotation_error_deg_rmse']:.2f}")
        print(f"  Range: [{overall_stats['rotation_error_deg_min']:.2f}, {overall_stats['rotation_error_deg_max']:.2f}]")
        
        # Per-axis translation errors
        print(f"\nTranslation Error by Axis (m):")
        for axis in ['x', 'y', 'z']:
            key = f'translation_error_{axis}_m'
            print(f"  {axis.upper()}: {overall_stats[f'{key}_mean']:.4f} ± {overall_stats[f'{key}_std']:.4f}")
        
        # Per-axis rotation errors  
        print(f"\nRotation Error by Axis (deg):")
        for axis in ['x', 'y', 'z']:
            key = f'rotation_error_{axis}_deg'
            print(f"  {axis.upper()}: {overall_stats[f'{key}_mean']:.2f} ± {overall_stats[f'{key}_std']:.2f}")
        
        # Per-camera statistics
        print(f"\nPER-CAMERA STATISTICS:")
        print("-" * 60)
        for cam_name in self.trailer_cameras:
            cam_stats = self.compute_statistics(camera_errors[cam_name])
            if cam_stats:
                print(f"\n{cam_name} ({len(camera_errors[cam_name])} poses):")
                print(f"  Translation RMSE: {cam_stats['translation_error_m_rmse']:.4f} m")
                print(f"  Rotation RMSE: {cam_stats['rotation_error_deg_rmse']:.2f} deg")
        
        # Save detailed results
        self.save_detailed_results(all_errors)
    
    def save_detailed_results(self, all_errors: List[Dict]):
        """Save detailed results to CSV file."""
        if not all_errors:
            return
        
        output_file = self.version_path / 'vggt_evaluation_results.csv'
        df = pd.DataFrame(all_errors)
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGGT calibration results')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root path to the STT4AT dataset')
    
    args = parser.parse_args()
    
    print("=== VGGT Calibration Evaluation ===")
    print(f"Data root: {args.data_root}")
    
    evaluator = VGGTCalibrationEvaluator(args.data_root)
    results = evaluator.evaluate_calibration()
    evaluator.print_results(results)


if __name__ == '__main__':
    main() 
