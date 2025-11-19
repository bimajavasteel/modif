import threading
from typing import Any, Optional, List
import insightface

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = [None, None]  # Dual GPU support
THREAD_LOCK = threading.Lock()


def get_face_analyser(device_id: int = 0) -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER[device_id] is None:
            providers = [('CUDAExecutionProvider', {'device_id': device_id})]
            FACE_ANALYSER[device_id] = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            FACE_ANALYSER[device_id].prepare(ctx_id=0)
    return FACE_ANALYSER[device_id]


def clear_face_analyser() -> None:
    global FACE_ANALYSER
    FACE_ANALYSER = [None, None]


def get_one_face(frame: Frame, position: int = 0, device_id: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame, device_id)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_many_faces(frame: Frame, device_id: int = 0) -> Optional[List[Face]]:
    if frame is None or frame.size == 0:
        return None
    try:
        return get_face_analyser(device_id).get(frame)
    except (ValueError, RuntimeError) as e:
        print(f"[FaceAnalyser] Skipped invalid frame: {e}")
        return None


def find_similar_face(frame: Frame, reference_face: Face, device_id: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame, device_id)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = sum((a - b) ** 2 for a, b in zip(face.normed_embedding, reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
