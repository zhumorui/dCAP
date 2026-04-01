#!/usr/bin/env python3
"""
Analyze VGGT calibration performance by scene to identify worst performing scenes.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re


def extract_scene_from_token(token: str) -> str:
    """Extract scene ID from token path."""
    # Token format: ../data/2025_07_20_16_43_59/049260_trailer_back.png
    # Extract the datetime part as scene ID
    match = re.search(r'/(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})/', token)
    if match:
        return match.group(1)
    else:
        # Fallback: use the directory name
        parts = token.split('/')
        if len(parts) >= 3:
            return parts[-2]
        return "unknown"


def analyze_scene_performance(csv_path: str):
    """Analyze performance by scene."""
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract scene from token
    df['scene'] = df['token'].apply(extract_scene_from_token)
    
    print(f"Total data points: {len(df)}")
    print(f"Unique scenes: {df['scene'].nunique()}")
    print(f"Unique cameras: {df['camera'].nunique()}")
    
    # Group by scene and compute statistics
    scene_stats = df.groupby('scene').agg({
        'translation_error_m': ['count', 'mean', 'std', 'median', 'max'],
        'rotation_error_deg': ['mean', 'std', 'median', 'max'],
        'translation_error_x_m': ['mean', 'std'],
        'translation_error_y_m': ['mean', 'std'],
        'translation_error_z_m': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    scene_stats.columns = ['_'.join(col).strip() for col in scene_stats.columns]
    
    # Sort by translation error mean (descending)
    scene_stats = scene_stats.sort_values('translation_error_m_mean', ascending=False)
    
    print("\n" + "="*100)
    print("SCENE PERFORMANCE RANKING (Worst to Best)")
    print("="*100)
    
    print(f"{'Rank':<4} {'Scene':<20} {'Count':<6} {'Trans_Mean':<10} {'Trans_Max':<10} {'Rot_Mean':<10} {'Rot_Max':<10}")
    print("-" * 100)
    
    for i, (scene, stats) in enumerate(scene_stats.iterrows(), 1):
        print(f"{i:<4} {scene:<20} {int(stats['translation_error_m_count']):<6} "
              f"{stats['translation_error_m_mean']:<10.4f} {stats['translation_error_m_max']:<10.4f} "
              f"{stats['rotation_error_deg_mean']:<10.2f} {stats['rotation_error_deg_max']:<10.2f}")
    
    # Show worst 5 scenes in detail
    print(f"\n{'='*100}")
    print("TOP 5 WORST PERFORMING SCENES - DETAILED ANALYSIS")
    print("="*100)
    
    worst_scenes = scene_stats.head(5)
    for scene, stats in worst_scenes.iterrows():
        print(f"\n🔴 SCENE: {scene}")
        print(f"   Data points: {int(stats['translation_error_m_count'])}")
        print(f"   Translation Error (m): Mean={stats['translation_error_m_mean']:.4f} ± {stats['translation_error_m_std']:.4f}, "
              f"Median={stats['translation_error_m_median']:.4f}, Max={stats['translation_error_m_max']:.4f}")
        print(f"   Rotation Error (deg): Mean={stats['rotation_error_deg_mean']:.2f} ± {stats['rotation_error_deg_std']:.2f}, "
              f"Median={stats['rotation_error_deg_median']:.2f}, Max={stats['rotation_error_deg_max']:.2f}")
        print(f"   Per-axis Trans Error (m): X={stats['translation_error_x_m_mean']:.4f}±{stats['translation_error_x_m_std']:.4f}, "
              f"Y={stats['translation_error_y_m_mean']:.4f}±{stats['translation_error_y_m_std']:.4f}, "
              f"Z={stats['translation_error_z_m_mean']:.4f}±{stats['translation_error_z_m_std']:.4f}")
    
    # Show best 5 scenes for comparison
    print(f"\n{'='*100}")
    print("TOP 5 BEST PERFORMING SCENES - FOR COMPARISON")
    print("="*100)
    
    best_scenes = scene_stats.tail(5)
    for scene, stats in best_scenes.iterrows():
        print(f"\n🟢 SCENE: {scene}")
        print(f"   Data points: {int(stats['translation_error_m_count'])}")
        print(f"   Translation Error (m): Mean={stats['translation_error_m_mean']:.4f} ± {stats['translation_error_m_std']:.4f}")
        print(f"   Rotation Error (deg): Mean={stats['rotation_error_deg_mean']:.2f} ± {stats['rotation_error_deg_std']:.2f}")
    
    # Analyze error patterns
    print(f"\n{'='*100}")
    print("ERROR PATTERN ANALYSIS")
    print("="*100)
    
    # Scenes with high variance
    high_variance_scenes = scene_stats[scene_stats['translation_error_m_std'] > 10].sort_values('translation_error_m_std', ascending=False)
    print(f"\nScenes with high translation error variance (std > 10m):")
    for scene, stats in high_variance_scenes.head(5).iterrows():
        print(f"  {scene}: std={stats['translation_error_m_std']:.4f}m, mean={stats['translation_error_m_mean']:.4f}m")
    
    # Scenes with consistently high errors
    consistent_high_error = scene_stats[
        (scene_stats['translation_error_m_median'] > 5) & 
        (scene_stats['translation_error_m_std'] < scene_stats['translation_error_m_mean'])
    ].sort_values('translation_error_m_median', ascending=False)
    
    if len(consistent_high_error) > 0:
        print(f"\nScenes with consistently high errors (median > 5m, low variance):")
        for scene, stats in consistent_high_error.head(5).iterrows():
            print(f"  {scene}: median={stats['translation_error_m_median']:.4f}m, mean={stats['translation_error_m_mean']:.4f}m")
    
    # Save detailed scene statistics
    output_path = Path(csv_path).parent / 'scene_performance_analysis.csv'
    scene_stats.to_csv(output_path)
    print(f"\nDetailed scene statistics saved to: {output_path}")
    
    return scene_stats


def main():
    parser = argparse.ArgumentParser(description='Analyze VGGT performance by scene')
    parser.add_argument('--csv-path', type=str, 
                       default='data/stt4at/v1.0-mini/vggt_evaluation_results.csv',
                       help='Path to VGGT evaluation results CSV')
    
    args = parser.parse_args()
    
    print("=== VGGT Scene Performance Analysis ===")
    analyze_scene_performance(args.csv_path)


if __name__ == '__main__':
    main()
