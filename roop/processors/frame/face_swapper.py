# ============================================================
#  face_swapper.py — Final Stable Multi-GPU Roop Patch (Dual GPU)
# ============================================================

from typing import Any, List, Callable, Tuple, Optional, Dict
import cv2
import insightface
import threading
import numpy as np
import time

from roop.gpu_parallel import start_gpu_workers, stop_gpu_workers
import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

NAME = "ROOP.FACE-SWAPPER-DUAL-GPU"
FACE_SWAPPER: Dict[int, Any] = {}
THREAD_LOCK = threading.Lock()
LANDMARK_FILTERS: Dict[str, Any] = {}

ONE_EURO_CONFIG = {
    "freq": 30.0,
    "min_cutoff": 1.0,
    "beta": 0.007,
    "d_cutoff": 1.0
}

class OneEuroFilter:
    def __init__(self, freq=30, min_cutoff=1, beta=0.0, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if t is not None and self.t_prev is not None:
            dt = max(t - self.t_prev, 1e-6)
            self.freq = 1.0 / dt
        self.t_prev = t if t else time.time()
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha_c = self.alpha(cutoff)
        x_hat = alpha_c * x + (1 - alpha_c) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

def get_face_swapper(device_id: int):
    global FACE_SWAPPER
    with THREAD_LOCK:
        if device_id not in FACE_SWAPPER or FACE_SWAPPER[device_id] is None:
            model_path = resolve_relative_path("../models/inswapper_128.onnx")
            providers = [("CUDAExecutionProvider", {"device_id": device_id})]
            FACE_SWAPPER[device_id] = insightface.model_zoo.get_model(model_path, providers=providers)
    return FACE_SWAPPER[device_id]

def clear_face_swapper():
    global FACE_SWAPPER
    FACE_SWAPPER = {}

def pre_check():
    download_directory_path = resolve_relative_path("../models")
    conditional_download(download_directory_path, ["https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx"])
    return True

def pre_start():
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False
    if not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status("No face in source path.", NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select image/video for target.", NAME)
        return False
    return True

def post_process():
    clear_face_swapper()
    clear_face_reference()
    LANDMARK_FILTERS.clear()

def safe_get_landmarks(face: Face):
    if face is None:
        return None
    for key in ["landmark_2d_106", "landmark_2d", "kps", "landmarks"]:
        if hasattr(face, key):
            data = getattr(face, key)
            if data is not None and len(data) > 0:
                return np.asarray(data, dtype=float)
    return None

def get_filter_for_key(key: str):
    if key not in LANDMARK_FILTERS:
        cfg = ONE_EURO_CONFIG
        LANDMARK_FILTERS[key] = OneEuroFilter(
            freq=cfg["freq"],
            min_cutoff=cfg["min_cutoff"],
            beta=cfg["beta"],
            d_cutoff=cfg["d_cutoff"]
        )
    return LANDMARK_FILTERS[key]

def smooth_landmarks(landmarks, key, timestamp):
    try:
        if landmarks is None:
            return landmarks
        f = get_filter_for_key(key)
        return f.filter(landmarks, timestamp)
    except:
        return landmarks

def robust_face_alignment(source_face, target_face, frame, device_id=0):
    try:
        sl = safe_get_landmarks(source_face)
        tl = safe_get_landmarks(target_face)
        if sl is None or tl is None:
            return frame, np.eye(2, 3, dtype=np.float32)
        timestamp = time.time()
        key_target = f"face_{getattr(target_face, 'face_index', 0)}" if roop.globals.many_faces else "reference"
        sl = smooth_landmarks(sl, "source", timestamp)
        tl = smooth_landmarks(tl, key_target, timestamp)
        if len(sl) < 3 or len(tl) < 3:
            return frame, np.eye(2, 3, dtype=np.float32)
        idx = list(range(min(5, len(sl))))
        sp = np.array([sl[i] for i in idx], dtype=np.float32)
        dp = np.array([tl[i] for i in idx], dtype=np.float32)
        mat = cv2.estimateAffinePartial2D(sp, dp, method=cv2.LMEDS)[0]
        if mat is None:
            return frame, np.eye(2, 3, dtype=np.float32)
        h, w = frame.shape[:2]
        aligned = cv2.warpAffine(frame, mat, (w, h))
        return aligned, mat
    except:
        return frame, np.eye(2, 3, dtype=np.float32)

def ensure_frame_format(f):
    if isinstance(f, np.ndarray):
        return f
    try:
        arr = np.array(f)
        return arr if arr.size > 0 else None
    except:
        return None

def swap_face_optimized(source_face, target_face, frame, device_id):
    try:
        aligned, _ = robust_face_alignment(source_face, target_face, frame, device_id)
        model = get_face_swapper(device_id)
        out = model.get(aligned, target_face, source_face, paste_back=False)
        out = ensure_frame_format(out)
        if out is None:
            return model.get(frame, target_face, source_face, paste_back=True)
        return out
    except:
        model = get_face_swapper(device_id)
        return model.get(frame, target_face, source_face, paste_back=True)

def process_frame_single_gpu(source_face, reference_face, frame, device_id):
    try:
        if roop.globals.many_faces:
            faces = get_many_faces(frame)
            if faces:
                for i, tf in enumerate(faces):
                    setattr(tf, "face_index", i)
                    frame = swap_face_optimized(source_face, tf, frame, device_id)
        else:
            tf = find_similar_face(frame, reference_face)
            if tf:
                frame = swap_face_optimized(source_face, tf, frame, device_id)
    except:
        pass
    return frame

def process_frames(source_path, temp_frame_paths, update):
    try:
        source_face = get_one_face(cv2.imread(source_path))
        reference_face = None if roop.globals.many_faces else get_face_reference()
        start_gpu_workers(process_frame_single_gpu)
        gpu_ids = roop.globals.gpu_device_ids or [0]
        idx = 0
        for f in temp_frame_paths:
            gpu = gpu_ids[idx]
            roop.globals.gpu_job_queue[gpu].put((f, source_face, reference_face))
            idx = (idx + 1) % len(gpu_ids)
        for gpu in gpu_ids:
            roop.globals.gpu_job_queue[gpu].join()
        stop_gpu_workers()
        if update:
            for _ in temp_frame_paths:
                update()
    except Exception as e:
        print("[MultiGPU] process_frames error:", e)

def process_image(source_path, target_path, out_path):
    try:
        src = get_one_face(cv2.imread(source_path))
        tgt = cv2.imread(target_path)
        ref = get_one_face(tgt)
        result = process_frame_single_gpu(src, ref, tgt, 0)
        cv2.imwrite(out_path, result)
    except:
        print("[face_swapper] process_image error")

def process_video(source_path, temp_frame_paths):
    try:
        if not roop.globals.many_faces and not get_face_reference():
            rf = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            set_face_reference(get_one_face(rf))
        roop.processors.frame.core.process_video(
            source_path,
            temp_frame_paths,
            process_frames
        )
    except Exception as e:
        print("[face_swapper] process_video error:", e)

_required = [
    "NAME",
    "pre_check",
    "pre_start",
    "process_frames",
    "process_image",
    "process_video",
    "post_process"
]

_missing = [_ for _ in _required if _ not in globals()]
if _missing:
    raise ImportError(f"[face_swapper] Missing API: {_missing}")

print(f"[face_swapper] LOADED - NAME={NAME} GPUs={roop.globals.gpu_device_ids}")
