# core/api.py
"""FastAPI application (shared between Modal and local)"""
import uuid
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json

from rs_common_interfaces_py import RsVideoCodec, RsVideoFormat, VideoConvertJob
from .storage import StorageBackend


def create_app(storage: StorageBackend, worker_func) -> FastAPI:
    """Create FastAPI app with injected storage and worker"""
    
    api = FastAPI(title="RS Video Converter API")
    
    @api.post("/submit")
    async def submit(job: VideoConvertJob):
        if not job.source or not job.source.url:
            raise HTTPException(status_code=400, detail="Missing 'url'")
        
        job_id = str(uuid.uuid4())
        storage.set_state(job_id, {
            "status": "queued",
            "progress": 0,
            "message": "Queued"
        })
        
        # Call worker (spawns on Modal, runs directly locally)
        worker_func(job_id, job)
        
        return {"job_id": job_id}
    
    @api.get("/status/{job_id}")
    async def status(job_id: str):
        data = storage.get_state(job_id)
        if not data:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return data
    
    @api.get("/download/{job_id}")
    async def download(job_id: str):
        data = storage.get_state(job_id)
        if not data or data.get("status") != "completed":
            raise HTTPException(status_code=404, detail="Not ready")
        
        path = data.get("file_path")
        name = data.get("file_name", "output.mp4")
        
        if not path or not storage.file_exists(path):
            raise HTTPException(status_code=404, detail="File missing")
        
        def file_iter():
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        mime = RsVideoFormat.from_filename(name).as_mime()
        headers = {"Content-Disposition": f"attachment; filename={name}"}
        return StreamingResponse(file_iter(), media_type=mime, headers=headers)
    
    @api.get("/progress/{job_id}/events")
    async def sse_progress(job_id: str):
        async def event_gen():
            sent_done = False
            while True:
                data = storage.get_state(job_id) or {"status": "unknown", "progress": 0}
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
                
                if data.get("status") in ("completed", "failed"):
                    if sent_done:
                        break
                    sent_done = True
                
                await asyncio.sleep(1)
        
        return StreamingResponse(event_gen(), media_type="text/event-stream")
    
    return api
