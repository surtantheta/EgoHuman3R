#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize EgoAllo outputs, rotating **only the scene point cloud**
by a fixed angle about the X axis (default +90°). We do NOT rotate any
poses (Ts_world_cpf, Ts_world_root) or local joint rotations. This avoids
twisting the human while letting you re-orient the scene.

Usage:
  python 4_visualize_outputs_scene_rx90.py \
    --search_root_dir /path/to/run \
    --rot_x_deg 90 \
    --recenter_floor_zero True
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Callable

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import tyro
import viser
from tqdm import tqdm
from projectaria_tools.core.data_provider import (
    VrsDataProvider,
    create_vrs_data_provider,
)
from projectaria_tools.core.sensor_data import TimeDomain

from egoallo import fncsmpl
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import InferenceTrajectoryPaths
from egoallo.network import EgoDenoiseTraj
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections


# ---------- helpers ----------

def _deg2rad(x: float) -> float:
    return x * np.pi / 180.0

def _Rx(deg: float, dtype=np.float32) -> np.ndarray:
    a = _deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c, -s],
        [0.0,  s,  c],
    ], dtype=dtype)

def _rotate_points_numpy_x(points_xyz: np.ndarray, rx_deg: float) -> np.ndarray:
    """Row-vector rotation: P' = P @ R^T, R is Rx(rx_deg)."""
    R = _Rx(rx_deg, dtype=points_xyz.dtype)
    return points_xyz @ R.T


# ---------- main ----------

def main(
    search_root_dir: Path,
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    rot_x_deg: float = 90.0,            # rotate scene about X by +90° by default
    recenter_floor_zero: bool = False,  # translate so rotated floor is z=0
) -> None:
    device = torch.device("cuda")
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    def get_file_list():
        return ["None"] + sorted(
            str(p.relative_to(search_root_dir))
            for p in search_root_dir.glob("**/egoallo_outputs/*.npz")
        )

    file_dropdown = server.gui.add_dropdown("File", options=get_file_list())

    @server.gui.add_button("Refresh File List").on_click
    def _(_evt):
        file_dropdown.options = get_file_list()

    traj_folder = server.gui.add_folder("Trajectory")
    current_file = "None"
    loop_cb: Callable[[], int] = lambda: 0

    while True:
        loop_cb()
        if current_file != file_dropdown.value:
            current_file = file_dropdown.value
            server.scene.reset()
            if current_file == "None":
                continue

            traj_folder.remove()
            traj_folder = server.gui.add_folder("Trajectory")

            with traj_folder:
                npz_path = Path(search_root_dir / current_file).resolve()
                loop_cb = load_and_visualize(
                    server=server,
                    npz_path=npz_path,
                    body_model=body_model,
                    device=device,
                    rot_x_deg=rot_x_deg,
                    recenter_floor_zero=recenter_floor_zero,
                )
                args_yaml = npz_path.parent / (npz_path.stem + "_args.yaml")
                if args_yaml.exists():
                    with server.gui.add_folder("Args"):
                        server.gui.add_markdown(f"```\n{args_yaml.read_text()}\n```")


def load_and_visualize(
    server: viser.ViserServer,
    npz_path: Path,
    body_model: fncsmpl.SmplhModel,
    device: torch.device,
    rot_x_deg: float,
    recenter_floor_zero: bool,
) -> Callable[[], int]:
    # ------- load outputs (NO rotations applied to poses) -------
    outputs = np.load(npz_path)
    timestamps_sec = outputs["timestamps_ns"] / 1e9
    (num_samples, timesteps, _, _) = outputs["body_quats"].shape

    # Paths & provider
    traj_dir = npz_path.resolve().parent.parent
    paths = InferenceTrajectoryPaths.find(traj_dir)
    provider = create_vrs_data_provider(str(paths.vrs_file))
    device_calib = provider.get_device_calibration()

    # World←CPF poses (keep as-is; do NOT rotate)
    Ts_world_cpf = torch.from_numpy(outputs["Ts_world_cpf"]).to(device)

    # Build Ts_world_device from stored poses (no extra rotation)
    T_device_cpf_np = device_calib.get_transform_device_cpf().to_quat_and_translation()
    T_device_cpf = SE3(torch.from_numpy(T_device_cpf_np).to(device=device, dtype=Ts_world_cpf.dtype))
    Ts_world_device = (SE3(Ts_world_cpf) @ T_device_cpf.inverse()).wxyz_xyz

    # Rotate only the sparse point cloud about X
    pc_raw, _ = load_point_cloud_and_find_ground(paths.points_path, "filtered")
    pts = pc_raw.xyz if hasattr(pc_raw, "xyz") else np.asarray(pc_raw)
    pts_rot = _rotate_points_numpy_x(pts, rot_x_deg)

    # Optional recenter: shift so floor is z=0 (scene-only)
    floor_z = float(np.percentile(pts_rot[:, 2], 0.5))
    if recenter_floor_zero:
        dz = -floor_z
        pts_rot = pts_rot + np.array([0.0, 0.0, dz], dtype=pts_rot.dtype)
        floor_z = 0.0

    print(f"[scene-rotate] Rx={rot_x_deg:.1f}° | floor_z={floor_z:.3f} m (poses unchanged)")

    # Detections (standard)
    hamer_detections = (
        CorrespondedHamerDetections.load(paths.hamer_outputs, timestamps_sec)
        if paths.hamer_outputs is not None else None
    )
    aria_detections = (
        CorrespondedAriaHandWristPoseDetections.load(
            paths.wrist_and_palm_poses_csv,
            timestamps_sec,
            Ts_world_device=Ts_world_device.detach().cpu().numpy(),
        )
        if paths.wrist_and_palm_poses_csv is not None else None
    )

    # Build traj: local joint rotations unchanged
    traj = EgoDenoiseTraj(
        betas=torch.from_numpy(outputs["betas"]).to(device),
        body_rotmats=SO3(torch.from_numpy(outputs["body_quats"])).as_matrix().to(device),
        contacts=torch.zeros((num_samples, timesteps, 21), device=device)
        if "contacts" not in outputs else torch.from_numpy(outputs["contacts"]).to(device),
        hand_rotmats=SO3(
            torch.from_numpy(
                np.concatenate([outputs["left_hand_quats"], outputs["right_hand_quats"]], axis=-2)
            ).to(device)
        ).as_matrix(),
    )

    # Ego video helper (unchanged)
    def get_ego_video(start_index: int, end_index: int, total_duration: float) -> bytes:
        assert isinstance(provider, VrsDataProvider)
        rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")
        assert rgb_stream_id is not None
        camera_fps = provider.get_configuration(rgb_stream_id).get_nominal_rate_hz()

        start_ns = int(outputs["timestamps_ns"][start_index])
        first_ns = provider.get_first_time_ns(rgb_stream_id, TimeDomain.RECORD_TIME)
        image_start_index = int((start_ns - first_ns) / 1e9 * camera_fps)
        image_end_index = min(
            int(image_start_index + (end_index - start_index) / 30.0 * camera_fps) + 5,
            provider.get_num_data(rgb_stream_id),
        )

        frames = []
        for i in tqdm(range(image_start_index, image_end_index)):
            image_data = provider.get_image_data_by_index(rgb_stream_id, i)[0]
            arr = image_data.to_numpy_array().copy()
            arr = cv2.resize(arr, (800, 800), interpolation=cv2.INTER_AREA)
            arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
            frames.append(arr)

        fps = len(frames) / total_duration
        output = io.BytesIO()
        iio.imwrite(output, frames, fps=fps, extension=".mp4",
                    codec="libx264", pixelformat="yuv420p", ffmpeg_params=["-crf", "23"])
        return output.getvalue()

    # Hand off to viewer — note we pass the rotated NumPy array for the scene
    return visualize_traj_and_hand_detections(
        server=server,
        Ts_world_cpf=Ts_world_cpf,   # unchanged (no twist)
        traj=traj,                   # local joints unchanged
        body_model=body_model,
        hamer_detections=hamer_detections,
        aria_detections=aria_detections,
        points_data=pts_rot,         # <-- rotated scene
        splat_path=paths.splat_path,
        floor_z=floor_z,
        get_ego_video=get_ego_video,
    )


if __name__ == "__main__":
    tyro.cli(main)
