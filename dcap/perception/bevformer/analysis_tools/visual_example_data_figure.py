"""Utility to visualise multi-camera images alongside a BEV overview for a custom sample."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from pyquaternion import Quaternion
from PIL import Image


DEFAULT_DATAROOT = "data/stt4at"
DEFAULT_VERSION = "v1.0-mini"
DEFAULT_SCENE_NAME = "2025_07_15_15_42_56"
DEFAULT_SAMPLE_TOKEN = "2025_07_15_15_42_56/029215"
DEFAULT_RANGE_METERS = 100.0

CAM_CHANNELS: Tuple[str, ...] = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render six camera views and a BEV plot (LiDAR + GT + sensor poses) for a nuScenes sample."
    )
    parser.add_argument("--dataroot", type=str, default=DEFAULT_DATAROOT, help="Path to nuScenes-format dataset root.")
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION, help="nuScenes schema version to load.")

    selector = parser.add_mutually_exclusive_group(required=False)
    selector.add_argument(
        "--sample-token",
        type=str,
        default=None,
        help=f"Sample token to visualise (defaults to {DEFAULT_SAMPLE_TOKEN!r} when not provided).",
    )
    selector.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help=f"Scene name to look up (defaults to {DEFAULT_SCENE_NAME!r} when selectors are omitted).",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index within the scene (ignored when sample token is provided).",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the rendered figure.")
    parser.add_argument(
        "--point-step",
        type=int,
        default=0,
        help="Subsample LiDAR points by this step size to keep plotting lightweight.",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=DEFAULT_RANGE_METERS,
        help="Evaluation range in meters (axes use range + 3 m to match visual.py).",
    )
    parser.add_argument(
        "--figure-size",
        type=float,
        nargs=2,
        default=(18.0, 8.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Matplotlib figure size in inches.",
    )
    parser.add_argument("--colormap", type=str, default="viridis", help="Colormap for LiDAR height rendering.")
    parser.add_argument(
        "--no-show", action="store_true", help="Disable plt.show(); useful when running headless."
    )
    parser.add_argument("--verbose", action="store_true", help="Print additional scene/sample information.")
    return parser.parse_args()


def resolve_sample(
    nusc: NuScenes, sample_token: Optional[str], scene_name: Optional[str], sample_index: int
) -> Dict:
    if sample_token:
        return nusc.get("sample", sample_token)

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")

    target_scene = next((scene for scene in nusc.scene if scene["name"] == scene_name), None)
    if target_scene is None:
        known = ", ".join(scene["name"] for scene in nusc.scene[:5])
        raise ValueError(f"Scene '{scene_name}' not found. Example names: {known}...")

    token = target_scene["first_sample_token"]
    for _ in range(sample_index):
        sample_rec = nusc.get("sample", token)
        if sample_rec["next"] == "":
            raise IndexError(
                f"Scene '{scene_name}' contains only {_ + 1} samples, cannot access index {sample_index}."
            )
        token = sample_rec["next"]

    sample_rec = nusc.get("sample", token)
    if sample_rec["scene_token"] != target_scene["token"]:
        raise RuntimeError("Sample traversal left the requested scene; dataset might be inconsistent.")
    return sample_rec


def load_camera_panels(axes: List[plt.Axes], nusc: NuScenes, sample: Dict) -> None:
    for ax, channel in zip(axes, CAM_CHANNELS):
        sample_data_token = sample["data"][channel]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(
            sample_data_token, box_vis_level=BoxVisibility.ANY
        )
        with Image.open(data_path) as img:
            width, height = img.size
            ax.imshow(np.asarray(img))

        for box in boxes:
            box.render(
                ax,
                view=camera_intrinsic,
                normalize=True,
                colors=("orange", "orange", "orange"),
                linewidth=1.0,
            )

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")


def lidar_points_in_sensor_frame(
    nusc: NuScenes, sample: Dict, point_step: int, nsweeps: int = 1
) -> Tuple[np.ndarray, Dict, Dict]:
    sd_token = sample["data"]["LIDAR_TOP"]
    sd_record = nusc.get("sample_data", sd_token)
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

    point_cloud, _ = LidarPointCloud.from_file_multisweep(
        nusc, sample, "LIDAR_TOP", "LIDAR_TOP", nsweeps=nsweeps
    )
    points = point_cloud.points[:3, ::point_step]
    return points, pose_record, cs_record


def gt_boxes_in_sensor_frame(
    nusc: NuScenes, sample: Dict, pose_record: Dict, lidar_cs_record: Dict
) -> List[Box]:
    boxes: List[Box] = []
    for ann_token in sample["anns"]:
        box = nusc.get_box(ann_token).copy()
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)
        box.translate(-np.array(lidar_cs_record["translation"]))
        box.rotate(Quaternion(lidar_cs_record["rotation"]).inverse)
        boxes.append(box)
    return boxes


def camera_poses_in_sensor_frame(
    nusc: NuScenes, sample: Dict, lidar_cs_record: Dict
) -> List[Tuple[str, np.ndarray, float]]:
    ref_from_car = transform_matrix(
        lidar_cs_record["translation"], Quaternion(lidar_cs_record["rotation"]), inverse=True
    )
    poses: List[Tuple[str, np.ndarray, float]] = []
    for channel in CAM_CHANNELS:
        sd_token = sample["data"][channel]
        sd_record = nusc.get("sample_data", sd_token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

        car_from_camera = transform_matrix(
            cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False
        )
        transform = ref_from_car @ car_from_camera

        position = transform @ np.array([0.0, 0.0, 0.0, 1.0])
        forward_vec = transform @ np.array([0.0, 0.0, 1.0, 0.0])
        yaw = float(np.arctan2(forward_vec[1], forward_vec[0]))
        poses.append((channel, position[:3], yaw))
    return poses


def draw_bev(
    ax: plt.Axes,
    lidar_points: np.ndarray,
    boxes: Iterable[Box],
    cameras: Iterable[Tuple[str, np.ndarray, float]],
    axes_limit: float,
    cmap: str,
    draw_pose: bool = True,
    draw_axes: bool = True,
    ego_axis_length: float = 4.0,
) -> None:
    forward = lidar_points[0, :]
    left = lidar_points[1, :]
    dists = np.sqrt(forward**2 + left**2)
    norm = np.clip(dists / max(axes_limit - 3.0, 1e-6), 0.0, 1.0)
    scatter = ax.scatter(forward, left, c=norm, s=0.5, cmap=cmap, alpha=0.6, linewidths=0)

    ax.plot(0.0, 0.0, "x", color="black", markersize=8, mew=2)
    for box in boxes:
        box.render(ax, view=np.eye(4), colors=("orange", "orange", "orange"), linewidth=1.0)

    if draw_pose:
        for channel, position, yaw in cameras:
            center = position[:2]
            base_triangle = np.array([[0.0, 0.0], [1.8, 0.7], [1.8, -0.7]])
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rotated = (rotation @ base_triangle.T).T + center
            polygon = Polygon(rotated, closed=True, facecolor="none", edgecolor="cyan", linewidth=1.0)
            ax.add_patch(polygon)

    if draw_axes:
        ax.quiver(
            0.0,
            0.0,
            ego_axis_length,
            0.0,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="red",
            width=0.005,
        )
        ax.quiver(
            0.0,
            0.0,
            0.0,
            ego_axis_length,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="green",
            width=0.005,
        )

    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()


def main() -> None:
    args = parse_args()
    if args.sample_token is None and args.scene_name is None:
        args.sample_token = DEFAULT_SAMPLE_TOKEN
        args.scene_name = DEFAULT_SCENE_NAME

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)
    sample = resolve_sample(nusc, args.sample_token, args.scene_name, args.sample_index)
    if args.verbose:
        print(f"Rendering sample {sample['token']} from scene {sample['scene_token']}.")

    lidar_points, pose_record, lidar_cs_record = lidar_points_in_sensor_frame(
        nusc, sample, max(args.point_step, 1)
    )
    boxes = gt_boxes_in_sensor_frame(nusc, sample, pose_record, lidar_cs_record)
    camera_pose_info = camera_poses_in_sensor_frame(nusc, sample, lidar_cs_record)

    side = min(args.figure_size)
    fig = plt.figure(figsize=(side, side))
    cell_w = 1.0 / 3.0
    sd_record = nusc.get("sample_data", sample["data"][CAM_CHANNELS[0]])
    cam_ratio = sd_record["height"] / sd_record["width"]
    bev_ratio = 1.0
    total_height_ratio = 2 * cam_ratio + bev_ratio

    fig_width = args.figure_size[0]
    fig_height = fig_width * total_height_ratio / 3.0
    fig = plt.figure(figsize=(fig_width, fig_height))

    row_heights = [
        cam_ratio / total_height_ratio,
        cam_ratio / total_height_ratio,
        bev_ratio / total_height_ratio,
    ]

    axes_matrix: List[List[plt.Axes]] = [[None] * 3 for _ in range(3)]
    top = 1.0
    for row_idx, row_height in enumerate(row_heights):
        bottom = top - row_height
        for col_idx in range(3):
            left = col_idx * cell_w
            ax = fig.add_axes([left, bottom, cell_w, row_height])
            axes_matrix[row_idx][col_idx] = ax
        top = bottom

    cam_axes: List[plt.Axes] = [
        axes_matrix[0][0],
        axes_matrix[0][1],
        axes_matrix[0][2],
        axes_matrix[1][0],
        axes_matrix[1][1],
        axes_matrix[1][2],
    ]
    load_camera_panels(cam_axes, nusc, sample)

    bev_ax_world = axes_matrix[2][0]
    draw_bev(
        axes_matrix[2][0],
        lidar_points,
        boxes,
        camera_pose_info,
        axes_limit=args.range + 3.0,
        cmap=args.colormap,
        draw_pose=False,
        draw_axes=False,
    )

    bev_ax_camera = axes_matrix[2][1]
    draw_bev(
        bev_ax_camera,
        lidar_points,
        boxes,
        camera_pose_info,
        axes_limit=20.0 + 3.0,
        cmap=args.colormap,
        draw_pose=True,
        draw_axes=True,
    )
    model_image_path = Path(__file__).resolve().parents[3] / "truck_trailer_model.png"
    if not model_image_path.exists():
        raise FileNotFoundError(f"Overlay image not found at {model_image_path}")
    with Image.open(model_image_path) as overlay_image:
        overlay_array = np.asarray(overlay_image)
    overlay_aspect = overlay_array.shape[0] / overlay_array.shape[1]
    overlay_half_width = 8.0 * 1.1
    overlay_half_height = overlay_half_width * overlay_aspect
    overlay_xmin = -overlay_half_width - 8.5
    overlay_xmax = overlay_half_width - 8.5
    overlay_ymin = -overlay_half_height + 0.4
    overlay_ymax = overlay_half_height + 0.4
    overlay_center_x = 0.5 * (overlay_xmin + overlay_xmax)
    overlay_center_y = 0.5 * (overlay_ymin + overlay_ymax)
    rotation_transform = Affine2D().rotate_deg_around(overlay_center_x, overlay_center_y, 2.0)
    bev_ax_camera.imshow(
        overlay_array,
        extent=(overlay_xmin, overlay_xmax, overlay_ymin, overlay_ymax),
        transform=rotation_transform + bev_ax_camera.transData,
        zorder=5,
    )

    map_ax = axes_matrix[2][2]
    map_image_path = Path(__file__).resolve().parents[3] / "town10HD_map_box.png"
    if not map_image_path.exists():
        raise FileNotFoundError(f"Map image not found at {map_image_path}")
    with Image.open(map_image_path) as map_image:
        map_array = np.asarray(map_image)
    map_ax.imshow(map_array)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    map_ax.axis("off")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight", dpi=200)
        if args.verbose:
            print(f"Saved figure to {output_path.resolve()}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
