#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Viz:
- Orange = P2 SMPL-X (bundle/smpl/*_verts_*.npz)  [pre-smoothed by EMA across frames per stable id/slot]
- Cyan   = P1 SMPL-H  (p1/mesh_*.obj, Umeyama-aligned to P2)
- FULL/CURRENT cloud toggle, strict per-frame removal (no mesh fallbacks).
- NEW: Follow frame camera toggle + one-shot Snap to frame camera button.
"""

import os, re, glob, argparse, time, inspect, math
import numpy as np

# ---------- tiny utils ----------
def npz_load(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}

def diag(msg): print(f"[DIAG] {msg}")

def read_tum_translations(txt_path):
    arr = []
    with open(txt_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            tok = ln.split()
            if len(tok) < 8: continue
            tx, ty, tz = map(float, tok[1:4])
            arr.append([tx, ty, tz])
    if not arr:
        raise ValueError(f"No valid translations parsed from {txt_path}")
    return np.asarray(arr, dtype=np.float64)

def umeyama_sim3(src_N3, dst_N3):
    src = np.asarray(src_N3, dtype=np.float64); dst = np.asarray(dst_N3, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0); mu_d = dst.mean(axis=0)
    X = src - mu_s; Y = dst - mu_d
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_s = (X**2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * (R @ mu_s)
    return float(s), R.astype(np.float64), t.astype(np.float64)

def apply_sim3_points(P, s, R, t):
    return (s * (P @ R.T)) + t

def list_p1_mesh_objs(p1_dir):
    paths = glob.glob(os.path.join(p1_dir, "mesh_*.obj"))
    pat = re.compile(r"mesh_(\d+)\.obj$")
    items = []
    for p in paths:
        m = pat.search(os.path.basename(p))
        if m: items.append((p, int(m.group(1))))
    items.sort(key=lambda x: x[1])
    return items

def load_obj_vertices_faces(path):
    vs, fs = [], []
    with open(path, "r") as f:
        for ln in f:
            if ln.startswith("v "):
                _, x,y,z = ln.strip().split()[:4]
                vs.append([float(x), float(y), float(z)])
            elif ln.startswith("f "):
                parts = ln.strip().split()[1:]
                idx = []
                for p in parts:
                    v = p.split("/")[0]
                    idx.append(int(v)-1)
                if len(idx) >= 3:
                    for k in range(1, len(idx)-1):
                        fs.append([idx[0], idx[k], idx[k+1]])
    V = np.asarray(vs, dtype=np.float32) if vs else np.empty((0,3), dtype=np.float32)
    F = np.asarray(fs, dtype=np.int32)   if fs else np.empty((0,3), dtype=np.int32)
    return V, F

# ---------- P2 bundle I/O ----------
def load_p2_bundle(bundle_dir):
    fdir = os.path.join(bundle_dir, "frames")
    cdir = os.path.join(bundle_dir, "camera")
    sdir = os.path.join(bundle_dir, "smpl")

    cam = npz_load(os.path.join(cdir, "camera.npz"))
    cam_dict = {"R": cam["R"].astype(np.float64),
                "t": cam["t"].astype(np.float64),
                "focal": cam["focal"], "pp": cam["pp"]}

    faces_path = os.path.join(sdir, "faces.npy")
    smplx_faces = np.load(faces_path) if os.path.exists(faces_path) else np.empty((0,3), np.int32)
    if smplx_faces.size == 0:
        diag("[WARN] smpl/faces.npy missing or empty → P2 orange meshes will not render.")

    frame_files = sorted(glob.glob(os.path.join(fdir, "*.npz")))
    pc_list, color_list, conf_list, msk_list = [], [], [], []
    for fp in frame_files:
        fr = npz_load(fp)
        pc_list.append(fr["pc"])        # (1,H,W,3)
        color_list.append(fr["color"])  # (1,H,W,3)
        conf_list.append(fr["conf"])    # (1,H,W)
        msk_list.append(fr["msk"])      # (1,H,W)

    verts_per_frame = []
    ids_per_frame = []
    for i in range(len(frame_files)):
        ids_fp = os.path.join(sdir, f"{i:06d}_ids.npz")
        meshes = []
        if os.path.exists(ids_fp):
            ids = npz_load(ids_fp).get("ids", np.empty((0,), dtype=np.int64))
            ids_per_frame.append(ids)
            k = 0
            while True:
                vfp = os.path.join(sdir, f"{i:06d}_verts_{k:02d}.npz")
                if not os.path.exists(vfp): break
                arr = npz_load(vfp)
                V = arr.get("verts", None)
                if V is not None:
                    V = np.asarray(V).astype(np.float32)
                    if V.ndim == 2 and V.shape[1] == 3:
                        meshes.append(V)
                    elif V.ndim == 3 and V.shape[-1] == 3:
                        for h in range(V.shape[0]): meshes.append(V[h].astype(np.float32))
                k += 1
        else:
            ids_per_frame.append(np.empty((0,), dtype=np.int64))
        verts_per_frame.append(meshes)

    return pc_list, color_list, conf_list, msk_list, cam_dict, smplx_faces, verts_per_frame, ids_per_frame

# ---------- pre-smoothing (EMA) ----------
def ema(prev, cur, beta):
    if prev is None or prev.shape != cur.shape:
        return cur.copy()
    b = float(np.clip(beta, 0.0, 0.99))
    return (b * prev) + ((1.0 - b) * cur)

def presmooth_p2_meshes(verts_per_frame, ids_per_frame, beta=0.6):
    caches = {}  # key -> prev verts
    out = []
    for f, meshes in enumerate(verts_per_frame):
        ids = ids_per_frame[f] if f < len(ids_per_frame) else np.empty((0,), dtype=np.int64)
        smoothed_meshes = []
        frame_keys = set()
        for i, V in enumerate(meshes):
            key = f"id{int(ids[i])}" if (i < len(ids) and ids.size > 0) else f"slot{i}"
            frame_keys.add(key)
            prev = caches.get(key, None)
            V_s = ema(prev, V, beta)
            caches[key] = V_s
            smoothed_meshes.append(V_s.astype(np.float32, copy=False))
        for k in list(caches.keys()):
            if k not in frame_keys:
                caches.pop(k, None)
        out.append(smoothed_meshes)
    return out

# ---------- camera helpers (follow/snap) ----------
def mat2quat(R: np.ndarray):
    # w, x, y, z (viser expects wxyz tuple)
    t = float(np.trace(R))
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        return (0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s)
    if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        s = 2.0*math.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2])
        return ((R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s)
    if R[1,1] > R[2,2]:
        s = 2.0*math.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2])
        return ((R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s)
    s = 2.0*math.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1])
    return ((R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s)

def set_all_clients_pose(server, R, t, look_at=None):
    q = mat2quat(R.astype(np.float32))
    pos = tuple(map(float, t))
    clients = server.get_clients()
    it = clients.values() if isinstance(clients, dict) else clients
    for c in it:
        c.camera.wxyz = q
        c.camera.position = pos
        if look_at is not None:
            c.camera.look_at = tuple(map(float, look_at))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Viz with pre-smoothed ORANGE meshes, strict removal, FULL/CURRENT cloud toggle, follow camera")
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--p1_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=500)
    ap.add_argument("--frame_offset", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--downsample_factor", type=int, default=16, help="Current-frame cloud stride.")
    ap.add_argument("--full_stride", type=int, default=8, help="FULL cloud stride.")
    ap.add_argument("--full_max_points_m", type=float, default=4.0, help="FULL cloud cap in millions.")
    ap.add_argument("--smooth_beta", type=float, default=0.8, help="EMA strength for ORANGE pre-smoothing.")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    # ---- P2 load ----
    (pc_list, color_list, conf_list, msk_list,
     cam_dict, smplx_faces, p2_verts_per_frame, p2_ids_per_frame) = load_p2_bundle(args.bundle_dir)

    # Trim to N
    N = min(args.num_frames, len(pc_list))
    if N <= 0:
        raise RuntimeError("No frames found in bundle.")
    pc_list            = pc_list[:N]
    color_list         = color_list[:N]
    conf_list          = conf_list[:N]
    msk_list           = msk_list[:N]
    p2_verts_per_frame = p2_verts_per_frame[:N]
    p2_ids_per_frame   = p2_ids_per_frame[:N]
    cam_dict = {
        "R": cam_dict["R"][:N],
        "t": cam_dict["t"][:N],
        "focal": cam_dict["focal"][:N],
        "pp": cam_dict["pp"][:N],
    }
    NB = N
    t_B = cam_dict["t"]
    diag(f"[bundle] frames_kept={NB}, frames_with_P2_mesh={sum(1 for ms in p2_verts_per_frame if len(ms)>0)}")

    # ---- pre-smooth ORANGE meshes once (offline) ----
    p2_verts_per_frame = presmooth_p2_meshes(p2_verts_per_frame, p2_ids_per_frame, beta=args.smooth_beta)

    # ---- P1 traj & pairing ----
    tum = os.path.join(args.p1_dir, "camera_cpf_poses.txt")
    if not os.path.isfile(tum): raise FileNotFoundError(f"Missing {tum}")
    p_A_full = read_tum_translations(tum)
    NA = p_A_full.shape[0]

    off = args.frame_offset
    if off >= 0:
        n_pair = min(NA, NB - off); idx_A = np.arange(n_pair); idx_B = np.arange(off, off + n_pair)
    else:
        n_pair = min(NA + off, NB);   idx_A = np.arange(-off, -off + n_pair); idx_B = np.arange(n_pair)
    if args.max_pairs is not None:
        n_pair = min(n_pair, args.max_pairs); idx_A = idx_A[:n_pair]; idx_B = idx_B[:n_pair]
    if n_pair < 3: raise ValueError("Too few paired frames after offset/max_pairs.")
    p_A = p_A_full[idx_A]; p_B = t_B[idx_B]

    # Sim(3) P1 -> P2
    s, R_um, t_um = umeyama_sim3(p_A, p_B)
    diag(f"[Sim3] s={s:.6f}\nR=\n{R_um}\nt={t_um}")

    # ---- P1 meshes transformed → P2 frame ----
    p1_items = list_p1_mesh_objs(args.p1_dir)
    p1_meshes_by_frame = {}  # frame -> [(V,F)]
    for path, idx in p1_items:
        V, F = load_obj_vertices_faces(path)
        if V.size == 0 or F.size == 0: continue
        V_tr = apply_sim3_points(V.astype(np.float64), s, R_um, t_um).astype(np.float32)
        b_idx = idx + off if off >= 0 else idx - (-off)
        if 0 <= b_idx < NB:
            p1_meshes_by_frame.setdefault(b_idx, []).append((V_tr, F))

    # ---------- FULL cloud (once) ----------
    def build_full_cloud():
        pts_all = []
        cols_all = []
        stride = max(1, int(args.full_stride))
        cap = int(args.full_max_points_m * 1_000_000)
        for i in range(NB):
            pc  = pc_list[i]; col = color_list[i]
            pts  = pc[0, ::stride, ::stride, :].reshape(-1, 3)
            cols = col[0, ::stride, ::stride, :].reshape(-1, 3)
            m = ~np.isnan(pts).any(axis=1)
            if m.any():
                pts_all.append(pts[m]); cols_all.append(cols[m])
            if sum(p.shape[0] for p in pts_all) >= cap:
                break
        if not pts_all:
            return np.zeros((0,3), np.float32), np.zeros((0,3), np.float32)
        P = np.concatenate(pts_all, axis=0).astype(np.float32)
        C = np.concatenate(cols_all, axis=0).astype(np.float32)
        if P.shape[0] > cap:
            idx = np.linspace(0, P.shape[0]-1, cap, dtype=np.int64)
            P = P[idx]; C = C[idx]
        diag(f"[full cloud] points={P.shape[0]} (cap={cap})")
        return P, C

    full_pts, full_cols = build_full_cloud()

    # ---------- Viser ----------
    import viser
    server = viser.ViserServer(host=args.host, port=args.port)
    diag(f"[viser] http://{args.host}:{args.port}")
    scene = server.scene

    # add_point_cloud helper (colors may be mandatory)
    _orig_apc = scene.add_point_cloud
    def _apc_compat(name, points, colors=None, **kwargs):
        if colors is None:
            colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.7
        return _orig_apc(name, points=points, colors=colors, **kwargs)
    scene.add_point_cloud = _apc_compat

    # Robust mesh adder: solid RGB only (avoid vertex_colors), returns handle or None
    def _add_mesh_strict(name, V, F, rgb3):
        if V.size == 0 or F.size == 0:
            return None
        V = np.ascontiguousarray(V, dtype=np.float32)
        F = np.ascontiguousarray(F, dtype=np.int32)
        if F.min() < 0 or (F.size > 0 and F.max() >= V.shape[0]):
            diag(f"[SKIP mesh] {name}: face index out of range"); return None
        rgb = np.asarray(rgb3, dtype=np.float32).reshape(-1)[:3]
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_tuple = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        add_simple = getattr(scene, "add_mesh_simple", None)
        add_mesh   = getattr(scene, "add_mesh", None)
        if add_simple is not None:
            try:
                return add_simple(name=name, vertices=V, faces=F, color=rgb_tuple,
                                  flat_shading=False, wireframe=False)
            except Exception as e:
                diag(f"[note] add_mesh_simple failed: {type(e).__name__}")
        if add_mesh is not None:
            try:
                return add_mesh(name=name, vertices=V, triangles=F, color=rgb_tuple)
            except Exception as e:
                diag(f"[note] add_mesh failed: {type(e).__name__}")
        diag(f"[FAIL mesh] {name}: mesh API unsupported on this viser build")
        return None

    ORANGE = (1.0, 0.55, 0.0)  # P2
    CYAN   = (0.0, 1.0, 1.0)   # P1

    # ---------- GUI ----------
    with server.gui.add_folder("Controls"):
        fslider = server.gui.add_slider("frame", min=0, max=NB-1, step=1, initial_value=0)
        show_cloud  = server.gui.add_checkbox("show_pointcloud", initial_value=True)
        cloud_mode  = server.gui.add_button_group("Cloud Mode", ("Current", "Full"))
        show_p2     = server.gui.add_checkbox("show_P2_meshes (orange)", initial_value=True)
        show_p1     = server.gui.add_checkbox("show_P1_meshes (cyan)", initial_value=True)
        ds = server.gui.add_slider("current_pc_stride", min=1, max=max(1, args.downsample_factor*2),
                                   step=1, initial_value=args.downsample_factor)
        psize = server.gui.add_slider("point_size", min=0.0001, max=0.05, step=0.0001, initial_value=0.004)
        follow_cam = server.gui.add_checkbox("Follow frame camera", initial_value=False)
        lock_up    = server.gui.add_checkbox("Lock up (+Z)", initial_value=False)
        snap_btn   = server.gui.add_button("Snap to frame camera")

    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Play", initial_value=False)
        fps     = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=12)

    # Try to lock/unlock up vector
    @lock_up.on_update
    def _(_evt):
        try:
            server.scene.set_up_direction("+z" if lock_up.value else None)
        except Exception:
            if lock_up.value:
                server.scene.set_up_direction("+z")

    # ---------- Live handles we remove every frame ----------
    prev_cloud_handle = None
    full_cloud_handle = None
    prev_p2_handles = []
    prev_p1_handles = []

    faces32 = smplx_faces.astype(np.int32) if smplx_faces.size > 0 else np.empty((0,3), np.int32)

    # build FULL cloud node once (keep hidden unless mode == Full)
    if full_pts.shape[0] > 0:
        full_cloud_handle = scene.add_point_cloud("/pc/full", points=full_pts, colors=full_cols, point_size=float(psize.value))
        full_cloud_handle.visible = False

    def _remove_handles(handles):
        for h in handles:
            try: h.remove()
            except Exception: pass
        handles.clear()

    def _remove_handle(h):
        if h is None: return None
        try: h.remove()
        except Exception: pass
        return None

    def apply_frame_camera_to_clients(frame_idx):
        R = cam_dict["R"][frame_idx].astype(np.float32)
        t = cam_dict["t"][frame_idx].astype(np.float32)
        set_all_clients_pose(server, R, t)

    def draw_frame(frame, stride, diag_counts=False):
        nonlocal prev_cloud_handle, full_cloud_handle
        with server.atomic():
            # Clear previous drawables
            prev_cloud_handle = _remove_handle(prev_cloud_handle)
            _remove_handles(prev_p2_handles)
            _remove_handles(prev_p1_handles)

            # Cloud
            if show_cloud.value:
                mode = cloud_mode.value or "Current"
                if mode == "Full":
                    if full_cloud_handle is not None:
                        full_cloud_handle.point_size = float(psize.value)
                        full_cloud_handle.visible = True
                else:
                    if full_cloud_handle is not None:
                        full_cloud_handle.visible = False
                    pc  = pc_list[frame]; col = color_list[frame]
                    pts  = pc[0, ::stride, ::stride, :].reshape(-1, 3)
                    cols = col[0, ::stride, ::stride, :].reshape(-1, 3)
                    m = ~np.isnan(pts).any(axis=1)
                    pts, cols = pts[m], cols[m]
                    if pts.size > 0:
                        prev_cloud_handle = scene.add_point_cloud("/pc/current",
                                                                  points=pts.astype(np.float32),
                                                                  colors=cols.astype(np.float32),
                                                                  point_size=float(psize.value))
            else:
                if full_cloud_handle is not None:
                    full_cloud_handle.visible = False

            # P2 meshes (ORANGE) — pre-smoothed
            if show_p2.value and faces32.size > 0:
                meshes = p2_verts_per_frame[frame]
                for i, V in enumerate(meshes):
                    h = _add_mesh_strict(f"/p2/current_{i}", V, faces32, ORANGE)
                    if h is not None:
                        prev_p2_handles.append(h)

            # P1 meshes (CYAN)
            if show_p1.value and frame in p1_meshes_by_frame:
                meshes = p1_meshes_by_frame[frame]
                for i, (Vh, Fh) in enumerate(meshes):
                    h = _add_mesh_strict(f"/p1/current_{i}", Vh, Fh.astype(np.int32), CYAN)
                    if h is not None:
                        prev_p1_handles.append(h)

            # Apply camera pose to all clients if requested
            if follow_cam.value:
                apply_frame_camera_to_clients(frame)

        if diag_counts:
            diag(f"[frame {frame}] P2_meshes={len(prev_p2_handles)}  P1_meshes={len(prev_p1_handles)}  faces_present={faces32.size>0}")

    # Initial draw
    draw_frame(0, max(1, ds.value), diag_counts=True)

    # Reactivity
    @fslider.on_update
    def _(_v): draw_frame(fslider.value, max(1, ds.value))
    @show_cloud.on_update
    def _(_v): draw_frame(fslider.value, max(1, ds.value))
    @cloud_mode.on_click
    def _(_v): draw_frame(fslider.value, max(1, ds.value))
    @show_p2.on_update
    def _(_v): draw_frame(fslider.value, max(1, ds.value))
    @show_p1.on_update
    def _(_v): draw_frame(fslider.value, max(1, ds.value))
    @ds.on_update
    def _(_v): draw_frame(fslider.value, max(1, ds.value))
    @psize.on_update
    def _(_v): draw_frame(fslider.value, max(1, ds.value))

    @snap_btn.on_click
    def _(_evt):  # one-shot snap
        apply_frame_camera_to_clients(fslider.value)

    # Playback loop
    stop_flag = {"stop": False}
    def _player():
        last = time.time()
        while not stop_flag["stop"]:
            if playing.value:
                target = 1.0 / max(1, fps.value)
                now = time.time()
                if now - last >= target:
                    fslider.value = (fslider.value + 1) % NB
                    last = now
                else:
                    time.sleep(max(0.0, target - (now - last)))
            else:
                time.sleep(0.05)
    import threading; threading.Thread(target=_player, daemon=True).start()

    # Keep alive
    if hasattr(server, "wait_forever"): server.wait_forever()
    elif hasattr(server, "run"):         server.run()
    else:
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt:
            pass
    stop_flag["stop"] = True

if __name__ == "__main__":
    main()
