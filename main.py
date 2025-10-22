# main.py
"""Modal deployment entry point"""
import modal
from core.storage import ModalStorage
from core.worker import transcode_video
from core.api import create_app

# Modal resources
image = (
    modal.Image.from_registry(
        "jrottenberg/ffmpeg:8.0-nvidia",
        add_python="3.12"
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("fastapi[standard]", "requests", "rs-common-interfaces-py==0.1.2")
    .add_local_python_source("core")
)

out_vol = modal.Volume.from_name("av1-output", create_if_missing=True)
progress_kv = modal.Dict.from_name("av1-progress", create_if_missing=True)

app = modal.App("av1-background-converter", image=image)

# Create storage backend for Modal
storage = ModalStorage(progress_kv, out_vol)

# Worker function (Modal-decorated)
@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol": out_vol},
    timeout=60 * 60 * 2,
)
def transcode_worker(job_id: str, job):
    """Modal worker wrapper"""
    transcode_video(job_id, job, storage, use_gpu=True)

# Cleanup job
@app.function(
    image=image,
    volumes={"/vol": out_vol},
    schedule=modal.Period(hours=6),
    timeout=60 * 10,
)
def cleanup_old_files():
    """Delete old undownloaded files"""
    import time
    
    RETENTION_HOURS = 24
    cutoff_time = time.time() - (RETENTION_HOURS * 3600)
    deleted_count = 0
    
    for job_id in storage.list_jobs():
        data = storage.get_state(job_id)
        if not data:
            continue
        
        status = data.get("status")
        created_at = data.get("created_at", 0)
        downloaded = data.get("downloaded", False)
        already_deleted = data.get("deleted", False)
        
        if (status == "completed" and not downloaded and
            not already_deleted and created_at < cutoff_time):
            file_path = data.get("file_path")
            if file_path and storage.file_exists(file_path):
                storage.delete_file(file_path)
                data["deleted"] = True
                data["deleted_at"] = time.time()
                storage.set_state(job_id, data)
                deleted_count += 1
    
    storage.commit()
    print(f"Cleanup complete. Deleted {deleted_count} files.")

# Create and expose FastAPI app
def worker_func(job_id, job):
    """Wrapper that spawns Modal function"""
    transcode_worker.spawn(job_id, job)

api = create_app(storage, worker_func)

@app.function(image=image, volumes={"/vol": out_vol})
@modal.asgi_app()
def fastapi_app():
    return api
