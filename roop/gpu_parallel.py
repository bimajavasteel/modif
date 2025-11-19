import threading
import queue
import cv2
import roop.globals

# Worker threads per GPU
gpu_workers = {}
gpu_running = {}
gpu_job_queue = roop.globals.gpu_job_queue
gpu_result_queue = roop.globals.gpu_result_queue


def gpu_worker(gpu_id, worker_fn):
    gpu_running[gpu_id] = True

    while gpu_running[gpu_id]:
        try:
            job = gpu_job_queue[gpu_id].get()
            if job is None:
                gpu_job_queue[gpu_id].task_done()
                continue

            frame_path, source_face, reference_face = job

            frame = cv2.imread(frame_path)
            if frame is None:
                gpu_job_queue[gpu_id].task_done()
                continue

            result = worker_fn(source_face, reference_face, frame, gpu_id)

            if result is not None:
                cv2.imwrite(frame_path, result)

            # Progress callback active
            if hasattr(roop.globals, "progress_update_callback"):
                try:
                    roop.globals.progress_update_callback()
                except:
                    pass

            gpu_job_queue[gpu_id].task_done()

        except Exception as e:
            print(f"[GPU-WORKER-{gpu_id}] ERROR:", e)
            gpu_job_queue[gpu_id].task_done()


def start_gpu_workers(worker_fn):
    for gpu_id in roop.globals.gpu_device_ids:
        if gpu_id not in gpu_workers:
            t = threading.Thread(target=gpu_worker, args=(gpu_id, worker_fn), daemon=True)
            gpu_workers[gpu_id] = t
            t.start()


def stop_gpu_workers():
    for gpu_id in gpu_workers:
        gpu_running[gpu_id] = False
        gpu_job_queue[gpu_id].put(None)
