# local_server.py
"""Local development server"""
import uvicorn
import threading
from core.storage import LocalStorage
from core.worker import transcode_video
from core.api import create_app

# Create local storage
storage = LocalStorage(base_dir="./local_data")

# Worker runs in background thread locally
def worker_func(job_id, job):
    """Run worker in background thread"""
    thread = threading.Thread(
        target=transcode_video,
        args=(job_id, job, storage),
        kwargs={"use_gpu": False},  # Set to True if you have local GPU
        daemon=True
    )
    thread.start()

# Create FastAPI app
app = create_app(storage, worker_func)

if __name__ == "__main__":
    print("Starting local server on http://localhost:8000")
    print("Data stored in ./local_data")
    uvicorn.run(app, host="0.0.0.0", port=8000)
