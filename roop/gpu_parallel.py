# gpu_parallel.py
import threading
import cv2
import roop.globals

def gpu_worker(gpu_id, face_processor_fn):
    job_q = roop.globals.gpu_job_queue[gpu_id]
    while True:
        item = job_q.get()
        if item is None:
            break
        frame_path, source_face, reference_face = item
        try:
            frame = cv2.imread(frame_path)
            if frame is not None:
                result = face_processor_fn(source_face, reference_face, frame, gpu_id)
                cv2.imwrite(frame_path, result)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error:", e)
        roop.globals.gpu_result_queue.put(frame_path)
        job_q.task_done()

def start_gpu_workers(face_processor_fn):
    for gpu_id in roop.globals.gpu_device_ids:
        t = threading.Thread(target=gpu_worker, args=(gpu_id, face_processor_fn), daemon=True)
        roop.globals.active_gpu_workers[gpu_id] = t
        t.start()

def stop_gpu_workers():
    for gpu_id in roop.globals.gpu_device_ids:
        roop.globals.gpu_job_queue[gpu_id].put(None)
    for t in roop.globals.active_gpu_workers.values():
        t.join()
