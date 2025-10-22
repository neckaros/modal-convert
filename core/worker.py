# core/worker.py
"""Core transcoding logic (shared between Modal and local)"""
import subprocess
import tempfile
import time
import shlex
from typing import Optional
import requests

from rs_common_interfaces_py import VideoConvertJob, RsVideoCodec
from .storage import StorageBackend


def transcode_video(
    job_id: str,
    job: VideoConvertJob,
    storage: StorageBackend,
    use_gpu: bool = True
):
    """Core transcoding function - works locally or on Modal"""
    
    # Update state
    storage.set_state(job_id, {
        "status": "downloading",
        "progress": 0,
        "message": "Downloading source"
    })
    
    # Set default codec
    if not job.request.codec:
        job.request.codec = RsVideoCodec.AV1
    
    print(f"Starting job {job_id}")
    print(f"Format: {job.request.format}, Codec: {job.request.codec}, CRF: {job.request.crf}")
    
    with tempfile.TemporaryDirectory() as tmp:
        # Download source
        src = f"{tmp}/input"
        with requests.get(job.source.url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        
        # Probe duration
        duration_s = probe_duration(src)
        
        # Get output path
        dst = storage.get_file_path(job_id, f"output{job.request.format.to_extension()}")
        
        # Build FFmpeg command
        cmd = build_ffmpeg_command(job, src, dst, use_gpu)
        
        print("FFmpeg command:", " ".join(shlex.quote(arg) for arg in cmd))
        
        # Update state
        storage.set_state(job_id, {
            "status": "encoding",
            "progress": 0,
            "message": "Encoding started"
        })
        
        # Run FFmpeg
        success = run_ffmpeg_with_progress(cmd, duration_s, job_id, storage)
        
        if success:
            storage.set_state(job_id, {
                "status": "completed",
                "progress": 100,
                "message": "Done",
                "file_path": dst,
                "file_name": f"output{job.request.format.to_extension()}",
            })
        else:
            storage.set_state(job_id, {
                "status": "failed",
                "progress": 0,
                "message": "Encoding failed",
            })


def probe_duration(path: str) -> Optional[float]:
    """Probe video duration using ffprobe"""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            text=True,
        ).strip()
        return float(out) if out else None
    except Exception:
        return None


def build_ffmpeg_command(
    job: VideoConvertJob,
    src: str,
    dst: str,
    use_gpu: bool
) -> list:
    """Build FFmpeg command based on job configuration"""
    
    if use_gpu and job.request.codec == RsVideoCodec.AV1:
        return [
            "ffmpeg", "-hide_banner", "-nostats", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", src,
            "-c:v", "av1_nvenc",
            "-preset", "p4",
            "-cq", str(job.request.crf or 32),
            "-b:v", "0",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            dst,
        ]
    elif use_gpu and job.request.codec == RsVideoCodec.H265:
        return [
            "ffmpeg", "-hide_banner", "-nostats", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", src,
            "-c:v", "hevc_nvenc",
            "-preset", "p4",
            "-cq", str(job.request.crf or 28),
            "-b:v", "0",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            dst,
        ]
    else:
        # CPU fallback
        return [
            "ffmpeg", "-hide_banner", "-nostats", "-y",
            "-i", src,
            "-c:v", "libsvtav1",
            "-crf", str(job.request.crf or 32),
            "-preset", "6",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            dst,
        ]


def run_ffmpeg_with_progress(
    cmd: list,
    duration_s: Optional[float],
    job_id: str,
    storage: StorageBackend
) -> bool:
    """Run FFmpeg and track progress"""
    
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
        stdout = proc.stdout
        if stdout is None:
            # No stdout to iterate (shouldn't happen when stdout=PIPE), wait for process and return result.
            ret = proc.wait()
            return ret == 0

        for line in stdout:
            line = line.strip()
            
            if line.startswith("out_time_ms="):
                try:
                    out_time_ms = float(line.split("=", 1)[1])
                    if duration_s and duration_s > 0:
                        pct = min(99, int((out_time_ms / (duration_s * 1000.0)) * 100))
                except Exception:
                    pass
            
            # Update progress periodically
            now = time.time()
            if now - last_emit > 0.5:
                storage.set_state(job_id, {
                    "status": "encoding",
                    "progress": pct,
                    "message": "Encoding in progress",
                })
                last_emit = now
        
        ret = proc.wait()
        return ret == 0
    except Exception as e:
        print(f"Error during encoding: {e}")
        return False
