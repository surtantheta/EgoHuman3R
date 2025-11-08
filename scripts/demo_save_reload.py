import os, glob, time, json, cv2, numpy as np, torch, argparse, tempfile, shutil, random
from typing import List, Dict, Any, Tuple
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import roma

random.seed(42)

# --------------------------- tiny IO helpers ---------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _np_savez(path: str, **kwargs):
    _ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **kwargs)

def _np_load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}

def _save_json(path: str, obj):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _shape(x): return tuple(x.shape) if hasattr(x, "shape") else None
def _diag(msg): print(f"[DIAG] {msg}")
def _diag_frame(tag, i, pc, col, conf, msk):
    print(f"[DIAG] {tag} frame={i:04d} pc{_shape(pc)} col{_shape(col)} conf{_shape(conf)} msk{_shape(msk)}")

# --------------------------- argparse ---------------------------

def parse_args():
    p = argparse.ArgumentParser("Save/Reload with mask resize + SMPL meshes from params")
    p.add_argument("--model_path", type=str, default="src/human3r.pth")
    p.add_argument("--seq_path", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--vis_threshold", type=float, default=1.5)
    p.add_argument("--msk_threshold", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default="./tmp")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--reset_interval", type=int, default=10_000_000)
    p.add_argument("--use_ttt3r", action="store_true", default=False)
    # viewer tweakables (forwarded into SceneHumanViewer)
    p.add_argument("--downsample_factor", type=int, default=10)
    p.add_argument("--smpl_downsample", type=int, default=1)
    p.add_argument("--camera_downsample", type=int, default=1)
    p.add_argument("--mask_morph", type=float, default=10.0)

    # bundle
    p.add_argument("--export_bundle", action="store_true")
    p.add_argument("--bundle_dir", type=str, default=None)
    p.add_argument("--viz_only", action="store_true")
    return p.parse_args()

# --------------------------- frames ---------------------------

def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(os.path.join(p, "*")))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened(): raise ValueError(f"Error opening video file {p}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        nF  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f" - Video FPS: {fps}, total frames: {nF}")
        img_paths, tmpdirname = [], tempfile.mkdtemp()
        for i in range(nF):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok: break
            fp = os.path.join(tmpdirname, f"frame_{i:06d}.jpg")
            cv2.imwrite(fp, frame); img_paths.append(fp)
        cap.release()
    return img_paths, tmpdirname

# --------------------------- inputs ---------------------------

def prepare_input(img_paths, size, img_res=None, reset_interval=10_000_000):
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu")

    views = []
    for i in range(len(images)):
        view = {
            "img": images[i]["img"],  # (1,3,H,W) [-1,1]
            "ray_map": torch.full((images[i]["img"].shape[0], 6, images[i]["img"].shape[-2], images[i]["img"].shape[-1]), torch.nan),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
        }
        if img_res is not None:
            view["img_mhmr"] = pad_image(view["img"], img_res)
            view["K_mhmr"] = K_mhmr
        views.append(view)
        if (i+1) % reset_interval == 0:
            overlap = deepcopy(view)
            overlap["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap)
    return views

# --------------------------- mask resize ---------------------------

def _resize_mask_to_1HW(msk_1HW: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
    """
    Resize (1,Hm,Wm)->(1,target_H,target_W) via nearest; returns float32 [0,1].
    Accepts already-matching size (no-op). If invalid, returns zeros of target size.
    """
    if msk_1HW is None:
        return torch.zeros(1, target_H, target_W, dtype=torch.float32)
    m = msk_1HW
    if not torch.is_tensor(m): m = torch.from_numpy(np.asarray(m))
    if m.ndim != 3 or m.shape[0] != 1:
        return torch.zeros(1, target_H, target_W, dtype=torch.float32)
    Hm, Wm = m.shape[1], m.shape[2]
    if (Hm, Wm) == (target_H, target_W):
        return m.to(dtype=torch.float32)
    m0 = m[0].detach().cpu().numpy()
    m0r = cv2.resize(m0, (target_W, target_H), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(m0r[None, ...].astype(np.float32))

# --------------------------- SMPL helpers ---------------------------

def _have_param_set(pred_f: Dict[str, Any]) -> bool:
    # requires at least shape + rotmat + transl
    return all(k in pred_f for k in ["smpl_shape", "smpl_rotmat", "smpl_transl"])

def _smpl_verts_world_from_params_for_frame(pred_f, intrinsics_3x3, pose_c2w_4x4, smpl_layer) -> np.ndarray:
    """
    Build SMPL meshes from parameters for one frame.
    Returns (Nh,V,3) in WORLD coords (np.float32), or empty (0,3) if Nh==0.
    Expected shapes per your snippet:
      smpl_shape:     (1,Nh,10)
      smpl_rotmat:    (1,Nh,53,3,3)   # 6D or 3x3 blocks; we convert to rotvec(53,3)
      smpl_transl:    (1,Nh,3)
      smpl_expression:(1,Nh,expr) or None
      smpl_id:        (1,Nh)
    """
    from src.dust3r.utils.geometry import geotrf

    # extract
    betas   = pred_f.get("smpl_shape", torch.empty(1,0,10))[0]       # (Nh,10)
    rotmat  = pred_f.get("smpl_rotmat", torch.empty(1,0,53,3,3))[0]  # (Nh,53,3,3)
    transl  = pred_f.get("smpl_transl", torch.empty(1,0,3))[0]       # (Nh,3)
    expr    = pred_f.get("smpl_expression", [None])[0]               # (Nh,expr) or None

    Nh = betas.shape[0]
    if Nh == 0:
        return np.empty((0,3), dtype=np.float32)

    # smpl_layer API wants axis-angle (rotvec) for 53 joints (per your earlier script),
    # so convert the per-joint 3x3 rotmats to rotvecs:
    # rotmat: (Nh,53,3,3) -> rotvec: (Nh,53,3)
    rotvec = roma.rotmat_to_rotvec(rotmat)  # torch.float32

    with torch.no_grad():
        out = smpl_layer(
            rotvec,             # (Nh,53,3) axis-angle
            betas,              # (Nh,10)
            transl,             # (Nh,3)
            None, None,
            K=intrinsics_3x3.expand(Nh, -1, -1),  # (Nh,3,3)
            expression=expr     # (Nh,expr) or None
        )
        v_cam = out['smpl_v3d']                 # (Nh,V,3) in CAMERA
        v_world = geotrf(pose_c2w_4x4, v_cam[None, ...])[0]  # (Nh,V,3)
    return v_world.detach().cpu().numpy().astype(np.float32)

def _maybe_get_smpl_verts_world(pred_f, pose_f) -> np.ndarray:
    """
    Try to read SMPL verts already provided by the model.
    Priority: explicit world keys; else camera verts -> world.
    """
    from src.dust3r.utils.geometry import geotrf

    def _to_np(x):
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return None

    for k in ["smpl_verts_world", "smpl_v3d_world", "smpl_vertices_world", "verts_world", "smplx_verts_world"]:
        if k in pred_f:
            arr = _to_np(pred_f[k][0] if (hasattr(pred_f[k], "ndim") and pred_f[k].ndim == 4) else pred_f[k])
            if arr is not None and arr.ndim == 3 and arr.shape[-1] == 3:
                return arr.astype(np.float32)

    for k in ["smpl_verts", "smpl_v3d", "smpl_vertices", "verts", "smplx_verts"]:
        if k in pred_f:
            arr = pred_f[k]
            arr_np = _to_np(arr[0] if (hasattr(arr, "ndim") and arr.ndim == 4) else arr)
            if arr_np is not None and arr_np.ndim == 3 and arr_np.shape[-1] == 3:
                vw = geotrf(pose_f, torch.from_numpy(arr_np)[None, ...])[0]
                return vw.detach().cpu().numpy().astype(np.float32)

    return np.empty((0,3), dtype=np.float32)

# --------------------------- outputs ---------------------------

def prepare_output(outputs):
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer

    # keep last pass
    valid_length = len(outputs["pred"])
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # drop overlaps due to reset
    reset_mask = torch.cat([v["reset"] for v in outputs["views"]], 0)
    shifted = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], 0)
    outputs["pred"]  = [p for p, m in zip(outputs["pred"],  shifted) if not m]
    outputs["views"] = [v for v, m in zip(outputs["views"], shifted) if not m]
    reset_mask = reset_mask[~shifted]

    pts3d_self_ls = [o["pts3d_in_self_view"] for o in outputs["pred"]]  # (1,H,W,3)
    conf_self     = [o["conf_self"] for o in outputs["pred"]]           # (1,H,W)
    pts3d_self    = torch.cat(pts3d_self_ls, 0)                          # (B,H,W,3)
    B, H, W, _ = pts3d_self.shape

    # camera poses (B,1,4,4-like list)
    pr_poses = [pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in outputs["pred"]]
    if reset_mask.any():
        pr_cat = torch.cat(pr_poses, 0)
        I = torch.eye(4, device=pr_cat.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_cat, I)
        bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([I.unsqueeze(0), bases[:-1]], 0)
        pr_cat = torch.einsum('bij,bjk->bik', shifted_bases, pr_cat)
        pr_poses = list(pr_cat.unsqueeze(1).unbind(0))

    # to world
    pts3d_world = [geotrf(pose, p_self.unsqueeze(0)) for pose, p_self in zip(pr_poses, pts3d_self)]
    conf_world  = conf_self

    # intrinsics from depth
    pp = torch.tensor([W // 2, H // 2], device=pts3d_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3d_self, pp, focal_mode="weiszfeld")
    colors = [0.5 * (out["img"].permute(0, 2, 3, 1) + 1.0) for out in outputs["views"]]  # (1,H,W,3)

    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": torch.cat([P[:, :3, :3] for P in pr_poses], 0).numpy(),
        "t": torch.cat([P[:, :3, 3] for P in pr_poses], 0).numpy(),
    }

    # masks → resize to (1,H,W)
    has_mask = "msk" in outputs["pred"][0]
    msks_resized = []
    if has_mask:
        raw_msks = [o["msk"][..., 0] for o in outputs["pred"]]  # (1,Hm,Wm)
        for i in range(B):
            msks_resized.append(_resize_mask_to_1HW(raw_msks[i], H, W))
    else:
        msks_resized = [torch.zeros(1, H, W) for _ in range(B)]

    # ---------------- SMPL verts (prefer provided; else build from params) ----------------
    # Prepare intrinsics (B,3,3) from focal/pp for the SMPL_Layer call
    intrins_b = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    intrins_b[:, 0, 0] = focal.detach()
    intrins_b[:, 1, 1] = focal.detach()
    intrins_b[:, 0, 2] = pp[:, 0]
    intrins_b[:, 1, 2] = pp[:, 1]

    # Initialize SMPL layer once
    # Try to read num_betas from first available frame; default to 10
    _first = outputs["pred"][0]
    nb = int(_first.get("smpl_shape", torch.empty(1,0,10))[0].shape[-1]) if "smpl_shape" in _first else 10
    smpl_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=nb, kid=False, person_center='head')

    verts_list_per_frame: List[List[np.ndarray]] = []
    smpl_ids_list: List[torch.Tensor] = []

    # Diagnostics: print keys in first pred
    _diag(f"pred[0] keys: {sorted(list(outputs['pred'][0].keys()))}")

    for f in range(B):
        pred_f = outputs["pred"][f]
        pose_f = pr_poses[f]                         # (1,4,4)
        intr_f = intrins_b[f]                        # (3,3)

        # ID list (fallback 0..Nh-1)
        if "smpl_id" in pred_f:
            ids = pred_f["smpl_id"][0]
            if torch.is_tensor(ids): ids = ids.to(dtype=torch.int64)
        else:
            ids = None

        used = None

        # 1) use precomputed verts if present
        verts_world = _maybe_get_smpl_verts_world(pred_f, pose_f)
        if verts_world.size > 0:
            used = "precomputed-verts"
        else:
            # 2) if full param set exists, build via SMPL_Layer
            if _have_param_set(pred_f):
                Nh = pred_f["smpl_shape"][0].shape[0]
                if ids is None: ids = torch.arange(Nh, dtype=torch.int64)
                verts_world = _smpl_verts_world_from_params_for_frame(pred_f, intr_f, pose_f, smpl_layer)
                used = "params"
            else:
                verts_world = np.empty((0,3), dtype=np.float32)
                if ids is None: ids = torch.empty(0, dtype=torch.int64)
                used = "none"

        # to list of meshes
        if verts_world.ndim == 3 and verts_world.shape[-1] == 3:
            Nh = verts_world.shape[0]
            verts_list_per_frame.append([verts_world[i] for i in range(Nh)])
            if torch.is_tensor(ids):
                smpl_ids_list.append(ids)
            else:
                smpl_ids_list.append(torch.from_numpy(np.asarray(ids, dtype=np.int64)))
        else:
            verts_list_per_frame.append([])
            smpl_ids_list.append(torch.empty(0, dtype=torch.int64))

        _diag(f"[SMPL] frame {f:04d}: mode={used} Nh={len(verts_list_per_frame[-1])}")

    # final lists for viewer (numpy)
    pc_list    = [p.detach().cpu().numpy().astype(np.float32) for p in pts3d_world]  # (1,H,W,3)
    color_list = [c.detach().cpu().numpy().astype(np.float32) for c in colors]       # (1,H,W,3)
    conf_list  = [c.detach().cpu().numpy().astype(np.float32) for c in conf_world]   # (1,H,W)
    msk_list   = [m.detach().cpu().numpy().astype(np.float32) for m in msks_resized] # (1,H,W)

    # Faces: SMPL-X faces available on layer
    smpl_faces = smpl_layer.bm_x.faces.astype(np.int32) if hasattr(smpl_layer, "bm_x") else np.empty((0,3), dtype=np.int32)

    # Diagnostics
    for i in range(B):
        _diag_frame("pre-viewer", i, pc_list[i], color_list[i], conf_list[i], msk_list[i])
        _diag(f"  SMPL meshes: {len(verts_list_per_frame[i])}")

    return pc_list, color_list, conf_list, cam_dict, verts_list_per_frame, smpl_faces, smpl_ids_list, msk_list

# --------------------------- bundle save/load ---------------------------

def save_bundle(bundle_dir: str,
                pc_list: List[np.ndarray],
                color_list: List[np.ndarray],
                conf_list: List[np.ndarray],
                cam_dict: Dict[str, np.ndarray],
                verts_list: List[List[np.ndarray]],
                smpl_faces: np.ndarray,
                smpl_ids_list: List[torch.Tensor],
                msk_list: List[np.ndarray]) -> None:
    fdir = os.path.join(bundle_dir, "frames")
    cdir = os.path.join(bundle_dir, "camera")
    sdir = os.path.join(bundle_dir, "smpl")
    _ensure_dir(fdir); _ensure_dir(cdir); _ensure_dir(sdir)

    # cameras once
    _np_savez(os.path.join(cdir, "camera.npz"),
              R=cam_dict["R"], t=cam_dict["t"], focal=cam_dict["focal"], pp=cam_dict["pp"])

    # smpl faces (once)
    np.save(os.path.join(sdir, "faces.npy"), np.asarray(smpl_faces, dtype=np.int32))

    meta = []
    for i in range(len(pc_list)):
        pc, col, cf, mk = pc_list[i], color_list[i], conf_list[i], msk_list[i]
        _np_savez(os.path.join(fdir, f"{i:06d}.npz"), pc=pc, color=col, conf=cf, msk=mk)

        ids_t = smpl_ids_list[i]
        ids_np = ids_t.detach().cpu().numpy() if torch.is_tensor(ids_t) else np.asarray(ids_t)
        _np_savez(os.path.join(sdir, f"{i:06d}_ids.npz"), ids=ids_np)
        for k, vert in enumerate(verts_list[i]):
            _np_savez(os.path.join(sdir, f"{i:06d}_verts_{k:02d}.npz"), verts=vert)

        H, W = pc.shape[1], pc.shape[2]
        meta.append({"i": i, "H": int(H), "W": int(W), "Nh": int(len(verts_list[i]))})
    _save_json(os.path.join(bundle_dir, "manifest.json"),
               {"num_frames": len(pc_list), "frames": meta})
    print(f"[bundle] saved → {bundle_dir}")

def load_bundle(bundle_dir: str):
    fdir = os.path.join(bundle_dir, "frames")
    cdir = os.path.join(bundle_dir, "camera")
    sdir = os.path.join(bundle_dir, "smpl")

    cam_npz = _np_load(os.path.join(cdir, "camera.npz"))
    cam_dict = {"R": cam_npz["R"], "t": cam_npz["t"], "focal": cam_npz["focal"], "pp": cam_npz["pp"]}

    faces_path = os.path.join(sdir, "faces.npy")
    smpl_faces = np.load(faces_path) if os.path.exists(faces_path) else np.empty((0,3), dtype=np.int32)

    frame_files = sorted(glob.glob(os.path.join(fdir, "*.npz")))
    pc_list, color_list, conf_list, msk_list = [], [], [], []
    verts_list, ids_list = [], []

    for i, fp in enumerate(frame_files):
        fr = _np_load(fp)
        pc, col, cf, mk = fr["pc"], fr["color"], fr["conf"], fr["msk"]
        _diag_frame("load", i, pc, col, cf, mk)
        pc_list.append(pc); color_list.append(col); conf_list.append(cf); msk_list.append(mk)

        ids_fp = os.path.join(sdir, f"{i:06d}_ids.npz")
        frame_meshes = []
        if os.path.exists(ids_fp):
            ids_np = _np_load(ids_fp)["ids"]
            ids_list.append(torch.from_numpy(ids_np))
            k = 0
            while True:
                vfp = os.path.join(sdir, f"{i:06d}_verts_{k:02d}.npz")
                if not os.path.exists(vfp): break
                frame_meshes.append(_np_load(vfp)["verts"])
                k += 1
        else:
            ids_list.append(torch.empty(0, dtype=torch.int64))

        verts_list.append(frame_meshes)

    return pc_list, color_list, conf_list, cam_dict, verts_list, smpl_faces, ids_list, msk_list

# --------------------------- main paths ---------------------------

def run_inference(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available → CPU"); device = "cpu"

    add_path_to_dust3r(args.model_path)
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import SceneHumanViewer

    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths: print("No images found."); return
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]
    print(f"[info] using {len(img_paths)} frames")

    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device); model.eval()
    img_res = getattr(model, 'mhmr_img_res', None)

    views = prepare_input(img_paths, size=args.size, img_res=img_res, reset_interval=args.reset_interval)
    if tmpdirname is not None: shutil.rmtree(tmpdirname)

    print("[run] inference ...")
    t0 = time.time()
    outputs, _ = inference_recurrent_lighter(views, model, device, use_ttt3r=args.use_ttt3r)
    print(f"[run] done in {time.time()-t0:.2f}s")

    # build viewer payload (masks resized; SMPL meshes created from params if needed)
    pc_list, color_list, conf_list, cam_dict, verts_list, smpl_faces, smpl_id, msk_list = prepare_output(outputs)

    # bundle save
    bundle_dir = args.bundle_dir or os.path.join(args.output_dir, "output_bundle")
    if args.export_bundle:
        print(f"[bundle] exporting → {bundle_dir}")
        save_bundle(bundle_dir, pc_list, color_list, conf_list, cam_dict, verts_list, smpl_faces, smpl_id, msk_list)
    
    # viewer
    edge_colors = [None] * len(pc_list)
    viewer = SceneHumanViewer(
        pc_list=pc_list,
        color_list=color_list,
        conf_list=conf_list,
        cam_dict=cam_dict,
        all_smpl_verts=verts_list,   # list[list(V,3)]
        smpl_faces=smpl_faces,
        smpl_id=smpl_id,
        msk_list=msk_list,
        edge_color_list=edge_colors,
        device=device,
        show_camera=True,
        show_gt_camera=False,
        show_gt_smpl=False,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample
    )
    viewer.run()

def run_viz_only(args):
    from viser_utils import SceneHumanViewer
    bundle_dir = args.bundle_dir or os.path.join(args.output_dir, "output_bundle")
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"Bundle not found: {bundle_dir}")

    pc_list, color_list, conf_list, cam_dict, verts_list, smpl_faces, smpl_id, msk_list = load_bundle(bundle_dir)
    print(len(pc_list), len(color_list), len(conf_list), len(verts_list))
    pc_list = pc_list[:500]
    color_list = color_list[:500]
    conf_list = conf_list[:500]
    verts_list = verts_list[:500]

    edge_colors = [None] * len(pc_list)
    viewer = SceneHumanViewer(
        pc_list=pc_list,
        color_list=color_list,
        conf_list=conf_list,
        cam_dict=cam_dict,
        all_smpl_verts=verts_list,   # list[list(V,3)]
        smpl_faces=smpl_faces,
        smpl_id=smpl_id,
        msk_list=msk_list,
        edge_color_list=edge_colors,
        device=args.device,
        show_camera=True,
        show_gt_camera=False,
        show_gt_smpl=False,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample
    )
    viewer.run()

# --------------------------- entry ---------------------------

def main():
    args = parse_args()
    if args.viz_only:
        run_viz_only(args); return
    if not args.seq_path:
        print("Provide --seq_path or use --viz_only with --bundle_dir")
        return
    run_inference(args)

if __name__ == "__main__":
    main()
