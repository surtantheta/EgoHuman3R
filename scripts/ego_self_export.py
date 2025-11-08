from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.network import EgoDenoiseTraj
from egoallo.transforms import SE3, SO3


# ----------------- helpers -----------------

def _np(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

def _write_obj(path: Path, verts_xyz: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in verts_xyz:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

def _wxyz_to_R(wxyz: np.ndarray) -> np.ndarray:
    # normalized quaternion (w,x,y,z) -> 3x3
    w, x, y, z = wxyz
    n = np.linalg.norm([w, x, y, z]) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz],
    ], dtype=np.float32)

def _to_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T

def _write_tum(path: Path, Ts_world_cpf_7: np.ndarray, timestamps_sec: np.ndarray) -> None:
    # TUM: timestamp tx ty tz qx qy qz qw
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t, se3 in zip(timestamps_sec, Ts_world_cpf_7):
            w,x,y,z, px,py,pz = se3
            f.write(f"{t:.9f} {px:.6f} {py:.6f} {pz:.6f} {x:.6f} {y:.6f} {z:.6f} {w:.6f}\n")


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--smplh_npz", type=Path, default='./egoallo/data/smplh/neutral/model.npz')
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load EgoAllo outputs (exactly like viewer) ----
    out = np.load(args.npz)
    timestamps_sec = out["timestamps_ns"].astype(np.float64) / 1e9
    Ts_world_cpf = out["Ts_world_cpf"]                 # [T,7] (wxyz+xyz)
    body_quats   = out["body_quats"]                   # [S,T,21,4] (wxyz)
    left_quats   = out["left_hand_quats"]              # [S,T,15,4]
    right_quats  = out["right_hand_quats"]             # [S,T,15,4]
    betas_np     = out["betas"]                        # [S,B]

    S, T = body_quats.shape[0], body_quats.shape[1]
    sample = int(args.sample)
    step   = max(1, int(args.every))

    # Build EgoDenoiseTraj like visualizer
    body_rot = SO3(torch.from_numpy(body_quats)).as_matrix()
    hand_rot = SO3(torch.from_numpy(np.concatenate([left_quats, right_quats], axis=-2))).as_matrix()
    traj = EgoDenoiseTraj(
        betas=torch.from_numpy(betas_np).to(device),
        body_rotmats=body_rot.to(device),
        contacts=torch.zeros((S, T, 21), device=device),
        hand_rotmats=hand_rot.to(device),
    )

    # ---- body model & FK decomposition (same as viewer) ----
    body_model: fncsmpl.SmplhModel = fncsmpl.SmplhModel.load(args.smplh_npz).to(device)
    shaped = body_model.with_shape(torch.mean(traj.betas, dim=1, keepdim=True))

    body_quats_SO3 = SO3.from_matrix(traj.body_rotmats).wxyz
    if traj.hand_rotmats is not None:
        hquats = SO3.from_matrix(traj.hand_rotmats).wxyz
        lq = hquats[..., :15, :]
        rq = hquats[..., 15:, :]
    else:
        lq = rq = None

    fk = shaped.with_pose_decomposed(
        T_world_root=SE3.identity(device=device, dtype=body_quats_SO3.dtype).parameters(),
        body_quats=body_quats_SO3,
        left_hand_quats=lq,
        right_hand_quats=rq,
    )

    # Align to world exactly like the viewer
    T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        fk, torch.from_numpy(Ts_world_cpf).to(device)[None, ...]
    )
    fk = fk.with_new_T_world_root(T_world_root)

    # ---- data for LBS ----
    faces = _np(body_model.faces, np.int64)           # [F,3]
    W = _np(body_model.weights, np.float32)           # [V, J+1]
    verts_zero = _np(shaped.verts_zero, np.float32)   # [S,1,V,3]
    joints_zero = _np(shaped.joints_zero, np.float32) # [S,1,J,3]

    Twr = _np(fk.T_world_root, np.float32)            # [S,T,7]
    Twj = _np(fk.Ts_world_joint, np.float32)          # [S,T,J,7]
    J = Twj.shape[2]
    assert W.shape[1] == J + 1, f"weights second dim {W.shape[1]} != J+1={J+1}"

    # Build **bind** transforms at zero pose:
    #   root bind  : identity at origin
    #   joint bind : identity rotation, translation = joints_zero[s,0,j]
    bind_list = [np.eye(4, dtype=np.float32)]  # root
    j0 = joints_zero[sample, 0]                # [J,3]
    for j in range(J):
        bind_list.append(_to_4x4(np.eye(3, dtype=np.float32), j0[j]))
    T_bind = np.stack(bind_list, axis=0)       # [J+1,4,4]
    T_bind_inv = np.linalg.inv(T_bind)         # [J+1,4,4]

    # Prepare rest verts (homog)
    V = W.shape[0]
    v0 = verts_zero[sample, 0]                 # [V,3]
    v0_h = np.concatenate([v0, np.ones((V,1), dtype=np.float32)], axis=1)  # [V,4]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] sample={sample}, frames={T}, every={step}")

    for t in range(0, T, step):
        # World transforms this frame
        R_root = _wxyz_to_R(Twr[sample, t, :4])
        p_root = Twr[sample, t, 4:7]
        T_root = _to_4x4(R_root, p_root)

        T_list = [T_root]
        for j in range(J):
            Rj = _wxyz_to_R(Twj[sample, t, j, :4])
            pj = Twj[sample, t, j, 4:7]
            T_list.append(_to_4x4(Rj, pj))
        T_world = np.stack(T_list, axis=0)                       # [J+1,4,4]

        # Relative to bind: M_b = T_world,b @ inv(T_bind,b)
        M = T_world @ T_bind_inv                                 # [J+1,4,4]

        # Blend per vertex: Tv[v] = Σ_b w_vb * M_b
        Tv = np.einsum("vb,bkl->vkl", W, M, optimize=True)       # [V,4,4]

        v_h = (Tv @ v0_h[:, :, None])[:, :, 0]                   # [V,4]
        v = v_h[:, :3]                                           # [V,3]

        _write_obj(out_dir / f"mesh_{t:06d}.obj", v, faces)

    print(f"[export] meshes → {out_dir.resolve()}")

    # CPF camera in TUM format
    tum_path = out_dir / "camera_cpf_poses.txt"
    _write_tum(tum_path, _np(Ts_world_cpf, np.float64), timestamps_sec)
    print(f"[export] CPF camera poses → {tum_path.resolve()}")


if __name__ == "__main__":
    main()
