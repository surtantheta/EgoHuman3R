#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoAllo inference with a SINGLE fixed global rotation applied BEFORE sampling.

No RANSAC / IMU. You choose constant Euler angles (degrees) and we rotate:
  • Ts_world_cpf   (T+1)
  • Ts_world_device (T)
  • Sparse point cloud

Tip from your note: "drag cursor down by 90°" ≈ rotate world by +90° about X.
So try: --rot_x_deg 90 (you can also set Y/Z).

Example:
  python 3_aria_inference_fixedrot.py \
    --traj_root /path/to/run \
    --rot_x_deg 90 --rot_y_deg 0 --rot_z_deg 0 \
    --recenter_floor_zero True
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import viser
import yaml

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import (
    InferenceInputTransforms,
    InferenceTrajectoryPaths,
    load_denoiser,
)
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections


# ----------------------- helpers: rotations & point cloud -----------------------

def _deg2rad(x: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(x * np.pi / 180.0, device=device, dtype=dtype)

def _rx(deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = _deg2rad(deg, device, dtype)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([
        torch.stack([torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype), c, -s]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype), s,  c]),
    ])

def _ry(deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = _deg2rad(deg, device, dtype)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([
        torch.stack([ c, torch.tensor(0.0, device=device, dtype=dtype),  s]),
        torch.stack([ torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)]),
        torch.stack([-s, torch.tensor(0.0, device=device, dtype=dtype),  c]),
    ])

def _rz(deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = _deg2rad(deg, device, dtype)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([
        torch.stack([ c, -s, torch.tensor(0.0, device=device, dtype=dtype)]),
        torch.stack([ s,  c, torch.tensor(0.0, device=device, dtype=dtype)]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype),
                     torch.tensor(0.0, device=device, dtype=dtype),
                     torch.tensor(1.0, device=device, dtype=dtype)]),
    ])

def _compose_world_rotation(rx_deg: float, ry_deg: float, rz_deg: float,
                            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Compose fixed world rotation. We use R = Rz * Ry * Rx (left-multiplied to world->*)
    so set angles accordingly (all are in degrees).
    """
    Rx = _rx(rx_deg, device, dtype)
    Ry = _ry(ry_deg, device, dtype)
    Rz = _rz(rz_deg, device, dtype)
    return Rz @ Ry @ Rx  # (3,3)

def _rotate_points_numpy(points_xyz: np.ndarray, R: torch.Tensor) -> np.ndarray:
    """Row-vector rotation: P' = P @ R^T."""
    pts_t = torch.from_numpy(points_xyz).to(device=R.device, dtype=R.dtype)
    pts_rot = pts_t @ R.T
    return pts_rot.cpu().numpy()

def _wrap_points(points_data, new_xyz: np.ndarray):
    if hasattr(points_data, "xyz"):
        points_data.xyz = new_xyz
        return points_data
    out = SimpleNamespace(xyz=new_xyz)
    for attr in ("rgb", "colors", "color"):
        if hasattr(points_data, attr):
            setattr(out, attr, getattr(points_data, attr))
    return out


# ----------------------- main script -----------------------

@dataclasses.dataclass
class Args:
    traj_root: Path
    """
    traj_dir/
        video.vrs
        egoallo_outputs/
            {date}_{start_index}-{end_index}.npz
            ...
        ...
    """
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    # sampling window
    start_index: int = 0
    traj_length: int = 4000
    num_samples: int = 1

    # optional tiny CPF tilt (kept from your original)
    glasses_x_angle_offset: float = 0.0

    # NEW: fixed world rotation BEFORE sampling
    rot_x_deg: float = 90.0   # your note: drag down by 90° → about X
    rot_y_deg: float = 0.0
    rot_z_deg: float = 0.0

    # guidance/sampling config
    guidance_mode: GuidanceMode = "aria_hamer"
    guidance_inner: bool = True
    guidance_post: bool = True

    # IO
    save_traj: bool = True
    visualize_traj: bool = False
    recenter_floor_zero: bool = False  # translate so floor_z == 0 after rotation


def main(args: Args) -> None:
    device = torch.device("cuda")

    traj_paths = InferenceTrajectoryPaths.find(args.traj_root)
    if traj_paths.splat_path is not None:
        print("Found splat at", traj_paths.splat_path)
    else:
        print("No scene splat found.")

    # Load sparse point cloud (we will rotate this)
    points_data, _floor_z_initial = load_point_cloud_and_find_ground(traj_paths.points_path)

    # Read transforms from VRS / MPS, downsampled.
    transforms = InferenceInputTransforms.load(
        traj_paths.vrs_file, traj_paths.slam_root_dir, fps=30
    ).to(device=device)

    # Build Ts_world_cpf (T+1) and optional tiny x-tilt
    Ts_world_cpf = (
        SE3(
            transforms.Ts_world_cpf[
                args.start_index : args.start_index + args.traj_length + 1
            ]
        )
        @ SE3.from_rotation(
            SO3.from_x_radians(
                transforms.Ts_world_cpf.new_tensor(args.glasses_x_angle_offset)
            )
        )
    ).parameters()

    # Timestamps for detections align to T (drop the first)
    pose_timestamps_sec = transforms.pose_timesteps[
        args.start_index + 1 : args.start_index + args.traj_length + 1
    ]

    # Device poses (T) already sliced (drop first to match Ts_world_cpf[1:])
    Ts_world_device = transforms.Ts_world_device[
        args.start_index + 1 : args.start_index + args.traj_length + 1
    ]
    del transforms

    # ------------------ ONLY CHANGE: apply fixed global rotation ------------------
    R = _compose_world_rotation(
        args.rot_x_deg, args.rot_y_deg, args.rot_z_deg,
        device=Ts_world_cpf.device, dtype=Ts_world_cpf.dtype
    )  # (3,3)
    R_se3 = SE3.from_rotation(SO3.from_matrix(R))

    # rotate world->cpf (length T+1), device (length T), and point cloud
    Ts_world_cpf = (R_se3 @ SE3(Ts_world_cpf)).parameters()
    if Ts_world_device is not None and Ts_world_device.numel() != 0:
        Ts_world_device = (R_se3 @ SE3(Ts_world_device)).parameters()

    pts = getattr(points_data, "xyz", None)
    if pts is None:
        pts = np.asarray(points_data)
    pts_rot = _rotate_points_numpy(pts, R)
    points_data = _wrap_points(points_data, pts_rot)

    # robust floor from rotated cloud
    floor_z = float(np.percentile(points_data.xyz[:, 2], 0.5))
    print(f"[fixed-rotate] Rx={args.rot_x_deg:.1f}°, Ry={args.rot_y_deg:.1f}°, Rz={args.rot_z_deg:.1f}° | floor_z={floor_z:.3f} m")

    # Optional: translate so floor=0
    if args.recenter_floor_zero:
        t = torch.tensor([0.0, 0.0, -floor_z], device=Ts_world_cpf.device, dtype=Ts_world_cpf.dtype)
        T_shift = SE3.from_translation(t)
        Ts_world_cpf = (T_shift @ SE3(Ts_world_cpf)).parameters()
        if Ts_world_device is not None and Ts_world_device.numel() != 0:
            Ts_world_device = (T_shift @ SE3(Ts_world_device)).parameters()
        if hasattr(points_data, "xyz"):
            points_data.xyz = points_data.xyz + np.array([0.0, 0.0, -floor_z], dtype=points_data.xyz.dtype)
        else:
            points_data = (np.asarray(points_data) + np.array([0.0, 0.0, -floor_z])).astype(np.float32)
        floor_z = 0.0
        print("[recenter] floor_z set to 0.0 m")

    # ------------------ detections (unchanged) ------------------
    if traj_paths.hamer_outputs is not None:
        hamer_detections = CorrespondedHamerDetections.load(
            traj_paths.hamer_outputs,
            pose_timestamps_sec,
        ).to(device)
    else:
        print("No hand detections found.")
        hamer_detections = None

    if traj_paths.wrist_and_palm_poses_csv is not None:
        aria_detections = CorrespondedAriaHandWristPoseDetections.load(
            traj_paths.wrist_and_palm_poses_csv,
            pose_timestamps_sec,
            Ts_world_device=Ts_world_device.numpy(force=True) if Ts_world_device is not None else None,
        ).to(device)
    else:
        print("No Aria hand detections found.")
        aria_detections = None

    print(f"{Ts_world_cpf.shape=}")

    server = None
    if args.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)

    # ------------------ sampling (unchanged) ------------------
    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    traj = run_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode=args.guidance_mode,
        guidance_inner=args.guidance_inner,
        guidance_post=args.guidance_post,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=hamer_detections,
        aria_detections=aria_detections,
        num_samples=args.num_samples,
        device=device,
        floor_z=floor_z,
    )

    # ------------------ save (unchanged) ------------------
    if args.save_traj:
        save_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"_{args.start_index}-{args.start_index + args.traj_length}"
        )
        out_path = args.traj_root / "egoallo_outputs" / (save_name + ".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assert not out_path.exists()
        (args.traj_root / "egoallo_outputs" / (save_name + "_args.yaml")).write_text(
            yaml.dump(dataclasses.asdict(args))
        )

        posed = traj.apply_to_body(body_model)
        Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            posed, Ts_world_cpf[..., 1:, :]
        )
        print(f"Saving to {out_path}...", end="")
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
            Ts_world_root=Ts_world_root.numpy(force=True),
            body_quats=posed.local_quats[..., :21, :].numpy(force=True),
            left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
            right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
            contacts=traj.contacts.numpy(force=True),
            betas=traj.betas.numpy(force=True),
            frame_nums=np.arange(args.start_index, args.start_index + args.traj_length),
            timestamps_ns=(np.array(pose_timestamps_sec) * 1e9).astype(np.int64),
        )
        print("saved!")

    # ------------------ optional viz (unchanged) ------------------
    if args.visualize_traj:
        assert server is not None
        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf[1:],
            traj,
            body_model,
            hamer_detections,
            aria_detections,
            points_data=points_data,
            splat_path=traj_paths.splat_path,
            floor_z=floor_z,
        )
        while True:
            loop_cb()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
