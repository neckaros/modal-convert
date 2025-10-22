# modal_av1_service.py
import os
import uuid
import json
import time
import asyncio
import tempfile
import subprocess
import shlex
from typing import Optional

import modal
import requests
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# ---- Modal resources ----
image = (
    modal.Image.from_registry(
        "jrottenberg/ffmpeg:8.0-nvidia",  # Pre-built FFmpeg with NVENC
        add_python="3.12"
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("fastapi[standard]", "requests")
)

# Durable file storage for results
out_vol = modal.Volume.from_name("av1-output", create_if_missing=True)

# Shared state for progress/status keyed by job_id
progress_kv = modal.Dict.from_name("av1-progress", create_if_missing=True)

app = modal.App("av1-background-converter", image=image)

# ---- Worker: long-running job, spawned in background ----
@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol": out_vol},
    timeout=60 * 60 * 2,  # up to 1 hour per job; adjust as needed
)
def transcode_worker(job_id: str, url: str, vf: str = "av1", crf: int = 28):
    # Update initial state
    progress_kv[job_id] = {"status": "downloading", "progress": 0, "message": "Downloading source"}
    os.makedirs(f"/vol/{job_id}", exist_ok=True)
    print("Starting job", job_id, "with format", vf, "and CRF", crf)
    # Download input
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

        # Probe duration for progress %
        def probe_duration_seconds(path: str) -> Optional[float]:
            try:
                # ffprobe prints seconds as float on a single line
                out = subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        path,
                    ],
                    text=True,
                ).strip()
                return float(out) if out else None
            except Exception:
                return None

        duration_s = probe_duration_seconds(src)

        # Choose output container and path
        dst = f"/vol/{job_id}/output.mkv"

        # Build FFmpeg command based on encoder
        if vf == "av1":
            # Hardware AV1 encoding - minimal options for compatibility
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-y",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", src,
                "-c:v", "av1_nvenc",
                "-preset", "p4",  # p1 (fast) to p7 (slow/best), p4 is balanced
                "-cq", str(crf),  # Constant quality 0-51, lower=better (28-32 recommended)
                "-b:v", "0",  # Use CQ mode
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-movflags", "+faststart",
                "-progress", "pipe:1",
                dst,
            ]
        elif vf == "hevc":
            # Hardware HEVC encoding
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-y",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", src,
                "-c:v", "hevc_nvenc",
                "-preset", "p4",
                "-cq", str(crf),
                "-b:v", "0",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-movflags", "+faststart",
                "-progress", "pipe:1",
                dst,
            ]
        else:
            # CPU software encoding (fallback)
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-y",
                "-i", src,
                "-c:v", "libsvtav1",
                "-crf", str(crf),
                "-preset", "6",
                "-svtav1-params", "fast-decode=1",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-movflags", "+faststart",
                "-progress", "pipe:1",
                dst,
            ]
        print("\n=== FFmpeg Command ===")
        print(f"Command list: {cmd}")
        print(f"\nCommand as string: {' '.join(shlex.quote(arg) for arg in cmd)}")
        print(f"\nNumber of arguments: {len(cmd)}")
        progress_kv[job_id] = {"status": "encoding", "progress": 0, "message": "Encoding started"}

        # Run and parse progress
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_emit = time.time()
        pct = 0
        try:
            if proc.stdout is None:
                # If stdout is not available something went wrong creating the subprocess;
                # raise to ensure we surface the error instead of iterating over None.
                raise RuntimeError("ffmpeg subprocess has no stdout to read progress from")

            for line in proc.stdout:
                line = line.strip()
                # FFmpeg -progress emits out_time_ms / out_time among other keys
                if line.startswith("out_time_ms="):
                    try:
                        out_time_ms = float(line.split("=", 1)[1])
                        if duration_s and duration_s > 0:
                            pct = min(99, int((out_time_ms / (duration_s * 1000.0)) * 100))
                    except Exception:
                        pass

                # Occasionally publish progress to Dict
                now = time.time()
                if now - last_emit > 0.5:
                    progress_kv[job_id] = {
                        "status": "encoding",
                        "progress": pct,
                        "message": "Encoding in progress",
                    }
                    last_emit = now

            ret = proc.wait()
            if ret != 0:
                progress_kv[job_id] = {
                    "status": "failed",
                    "progress": pct,
                    "message": f"FFmpeg exited with code {ret}",
                }
                return

            # Done
            progress_kv[job_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Done",
                "file_path": dst,  # for download endpoint
                "file_name": "output.mkv",
            }
        except Exception as e:
            progress_kv[job_id] = {
                "status": "failed",
                "progress": pct,
                "message": f"Error: {e}",
            }



# ---- Cleanup function: delete file after download ----
def delete_file_task(job_id: str, file_path: str):
    """Background task to delete file from volume after download"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # Remove parent job directory if empty
            job_dir = os.path.dirname(file_path)
            if os.path.isdir(job_dir) and not os.listdir(job_dir):
                os.rmdir(job_dir)
            # Mark as deleted in metadata
            data = progress_kv.get(job_id, {})
            data["deleted"] = True
            data["deleted_at"] = time.time()
            progress_kv[job_id] = data
            print(f"Deleted file for job {job_id}")
    except Exception as e:
        print(f"Failed to delete file for job {job_id}: {e}")

# ---- Scheduled cleanup: remove old undownloaded files ----
@app.function(
    image=image,
    volumes={"/vol": out_vol},
    schedule=modal.Period(hours=6),  # Run every 6 hours
    timeout=60 * 10,
)
def cleanup_old_files():
    """Delete files older than RETENTION_HOURS that were never downloaded"""
    RETENTION_HOURS = 24  # Keep files for 24 hours if not downloaded
    
    print("Starting scheduled cleanup...")
    cutoff_time = time.time() - (RETENTION_HOURS * 3600)
    deleted_count = 0
    
    # Iterate through all jobs in Dict
    for job_id in progress_kv.keys():
        try:
            data = progress_kv.get(job_id)
            if not data:
                continue
            
            status = data.get("status")
            created_at = data.get("created_at", 0)
            downloaded = data.get("downloaded", False)
            already_deleted = data.get("deleted", False)
            
            # Delete if:
            # 1. Completed but never downloaded and older than retention period
            # 2. Not already deleted
            if (
                status == "completed"
                and not downloaded
                and not already_deleted
                and created_at < cutoff_time
            ):
                file_path = data.get("file_path")
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    job_dir = os.path.dirname(file_path)
                    if os.path.isdir(job_dir) and not os.listdir(job_dir):
                        os.rmdir(job_dir)
                    
                    data["deleted"] = True
                    data["deleted_at"] = time.time()
                    progress_kv[job_id] = data
                    
                    deleted_count += 1
                    age_hours = (time.time() - created_at) / 3600
                    print(f"Deleted old file for job {job_id} (age: {age_hours:.1f}h)")
        
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
    
    # Commit all deletions to volume
    out_vol.commit()
    print(f"Cleanup complete. Deleted {deleted_count} files.")



# ---- FastAPI app: submit, status, download, SSE ----
api = FastAPI()

@api.post("/submit")
async def submit(payload: dict = Body(...)):
    print("Submit payload:", payload)
    url = payload.get("url")
    print("Received submit request for URL:", url)
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url'")
    vf = payload.get("format", "av1")
    crf_value = payload.get("crf")
    if crf_value is not None:
        try:
            crf = int(crf_value)
        except ValueError:
            # Handle invalid (non-int) value, e.g., log error or use default
            print(f"Invalid CRF value '{crf_value}', using default 23")
            crf = 23
    else:
        crf = 23
    job_id = str(uuid.uuid4())
    progress_kv[job_id] = {"status": "queued", "progress": 0, "message": "Queued"}
    transcode_worker.spawn(job_id, url, vf=vf, crf=crf)  # background job
    return {"job_id": job_id}

@api.get("/status/{job_id}")
async def status(job_id: str):
    print("Status request for job_id:", job_id)
    data = progress_kv.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return data

@api.get("/download/{job_id}")
async def download(job_id: str):
    data = progress_kv.get(job_id)
    if not data or data.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Not ready")
    path = data.get("file_path")
    name = data.get("file_name", "output.mkv")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing")
    def file_iter():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        delete_file_task(job_id, path)
    headers = {"Content-Disposition": f"attachment; filename={name}"}
    return StreamingResponse(file_iter(), media_type="video/x-matroska", headers=headers)

@api.get("/progress/{job_id}/events")
async def sse_progress(job_id: str):
    async def event_gen():
        sent_done = False
        while True:
            data = progress_kv.get(job_id) or {"status": "unknown", "progress": 0}
            payload = json.dumps(data)
            yield f"data: {payload}\n\n"
            if data.get("status") in ("completed", "failed"):
                if sent_done:
                    break
                sent_done = True
            await asyncio.sleep(1)
    return StreamingResponse(event_gen(), media_type="text/event-stream")

# Expose FastAPI via Modal
@app.function(image=image, volumes={"/vol": out_vol})
@modal.asgi_app()
def fastapi_app():
    return api
