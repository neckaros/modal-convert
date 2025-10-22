# core/storage.py
"""Storage abstraction for Modal and local environments"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator
import os
import json
import tempfile
from pathlib import Path


class StorageBackend(ABC):
    """Abstract storage for state and files"""
    
    @abstractmethod
    def get_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job state"""
        pass
    
    @abstractmethod
    def set_state(self, job_id: str, data: Dict[str, Any]):
        """Set job state"""
        pass
    
    @abstractmethod
    def list_jobs(self) -> Iterator[str]:
        """List all job IDs"""
        pass
    
    @abstractmethod
    def get_file_path(self, job_id: str, filename: str) -> str:
        """Get path for output file"""
        pass
    
    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        pass
    
    @abstractmethod
    def delete_file(self, path: str):
        """Delete a file"""
        pass
    
    @abstractmethod
    def commit(self):
        """Commit changes (no-op for local)"""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage"""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.state_dir = self.base_dir / "state"
        self.files_dir = self.base_dir / "files"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
    
    def get_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        state_file = self.state_dir / f"{job_id}.json"
        if not state_file.exists():
            return None
        return json.loads(state_file.read_text())
    
    def set_state(self, job_id: str, data: Dict[str, Any]):
        state_file = self.state_dir / f"{job_id}.json"
        state_file.write_text(json.dumps(data))
    
    def list_jobs(self) -> Iterator[str]:
        for f in self.state_dir.glob("*.json"):
            yield f.stem
    
    def get_file_path(self, job_id: str, filename: str) -> str:
        job_dir = self.files_dir / job_id
        job_dir.mkdir(exist_ok=True)
        return str(job_dir / filename)
    
    def file_exists(self, path: str) -> bool:
        return Path(path).exists()
    
    def delete_file(self, path: str):
        p = Path(path)
        if p.exists():
            p.unlink()
            # Remove empty parent directory
            if p.parent.exists() and not any(p.parent.iterdir()):
                p.parent.rmdir()
    
    def commit(self):
        pass  # No-op for local storage


class ModalStorage(StorageBackend):
    """Modal Dict + Volume storage"""
    
    def __init__(self, dict_obj, volume_obj, volume_path: str = "/vol"):
        self.dict = dict_obj
        self.volume = volume_obj
        self.volume_path = volume_path
    
    def get_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.dict.get(job_id)
    
    def set_state(self, job_id: str, data: Dict[str, Any]):
        self.dict[job_id] = data
    
    def list_jobs(self) -> Iterator[str]:
        return iter(self.dict.keys())
    
    def get_file_path(self, job_id: str, filename: str) -> str:
        job_dir = f"{self.volume_path}/{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        return f"{job_dir}/{filename}"
    
    def file_exists(self, path: str) -> bool:
        return os.path.exists(path)
    
    def delete_file(self, path: str):
        if os.path.exists(path):
            os.remove(path)
            job_dir = os.path.dirname(path)
            if os.path.isdir(job_dir) and not os.listdir(job_dir):
                os.rmdir(job_dir)
    
    def commit(self):
        self.volume.commit()
