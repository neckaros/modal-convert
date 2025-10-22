from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
import re
from urllib.parse import unquote
from datetime import datetime  # If needed for expiration handling, but using float as in Rust


class RsRequestStatus(Enum):
    UNPROCESSED = "unprocessed"
    PROCESSED = "processed"
    NEED_PARSING = "needParsing"
    REQUIRE_ADD = "requireAdd"
    INTERMEDIATE = "intermediate"
    NEED_FILE_SELECTION = "needFileSelection"
    FINAL_PRIVATE = "finalPrivate"
    FINAL_PUBLIC = "finalPublic"

    @classmethod
    def default(cls):
        return cls.UNPROCESSED


# Placeholder enums for undefined types
class RsResolution(Enum):
    UNKNOWN = "unknown"
    # Add other variants as needed

    @classmethod
    def from_filename(cls, filename: str) -> 'RsResolution':
        # TODO: Implement parsing logic
        return cls.UNKNOWN


class RsVideoFormat(Enum):
    OTHER = "other"
    # Add other variants as needed

    @classmethod
    def from_filename(cls, filename: str) -> 'RsVideoFormat':
        # TODO: Implement parsing logic
        return cls.OTHER


class RsVideoCodec(Enum):
    UNKNOWN = "unknown"
    # Add other variants as needed

    @classmethod
    def from_filename(cls, filename: str) -> 'RsVideoCodec':
        # TODO: Implement parsing logic
        return cls.UNKNOWN


class RsAudio(BaseModel):
    # Placeholder: Define fields as needed
    pass

    @classmethod
    def list_from_filename(cls, filename: str) -> List['RsAudio']:
        # TODO: Implement parsing logic
        return []


# Placeholder for RsRequestFiles
class RsRequestFiles(BaseModel):
    # Assuming it has filename and parse_filename method
    filename: Optional[str] = None
    # Add other fields as needed

    def parse_filename(self):
        # TODO: Implement as in RsRequest.parse_filename but for file
        pass


class RsCookie(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    domain: str
    http_only: bool
    path: str
    secure: bool
    expiration: Optional[float] = None
    name: str
    value: str

    @classmethod
    def from_string(cls, line: str) -> 'RsCookie':
        try:
            splitted = iter(line.split(';'))
            domain = next(splitted)
            http_only_str = next(splitted)
            path = next(splitted)
            secure_str = next(splitted)
            expiration_str = next(splitted)
            name = next(splitted)
            value = next(splitted)

            http_only = http_only_str == "true"
            secure = secure_str == "true"
            expiration = None
            if expiration_str.strip():
                expiration = float(expiration_str)

            return cls(
                domain=domain,
                http_only=http_only,
                path=path,
                secure=secure,
                expiration=expiration,
                name=name,
                value=value
            )
        except StopIteration:
            raise ValueError(f"Unable to parse cookie string: {line}")
        except ValueError as e:
            raise ValueError(f"Parsing error in cookie string '{line}': {e}")

    def netscape(self) -> str:
        second = "TRUE" if self.domain.startswith('.') else "FALSE"
        secure_str = "TRUE" if self.secure else "FALSE"
        expiration_str = str(int(self.expiration)) if self.expiration is not None else ""
        return f"{self.domain}\t{second}\t{self.path}\t{secure_str}\t{expiration_str}\t{self.name}\t{self.value}"

    def header(self) -> str:
        return f"{self.name}={self.value}"


def header_value(cookies: List[RsCookie]) -> str:
    return "; ".join(cookie.header() for cookie in cookies)


def headers(cookies: List[RsCookie]) -> Tuple[str, str]:
    return ("cookie", header_value(cookies))


class RsRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    upload_id: Optional[str] = None
    url: str
    mime: Optional[str] = None
    size: Optional[int] = None
    filename: Optional[str] = None
    status: RsRequestStatus = Field(default_factory=RsRequestStatus.default)
    permanent: bool = False
    json_body: Optional[Dict[str, Any]] = None
    method: str = "GET"  # Assuming RsRequestMethod is string; adjust if enum
    referer: Optional[str] = None
    headers: Optional[List[Tuple[str, str]]] = None
    cookies: Optional[List[RsCookie]] = None
    files: Optional[List[RsRequestFiles]] = None
    selected_file: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    people: Optional[List[str]] = None
    albums: Optional[List[str]] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    language: Optional[str] = None
    resolution: Optional[RsResolution] = None
    video_format: Optional[RsVideoFormat] = None
    videocodec: Optional[RsVideoCodec] = None
    audio: Optional[List[RsAudio]] = None
    quality: Optional[int] = None
    ignore_origin_duplicate: bool = False

    def set_cookies(self, cookies: List[RsCookie]):
        existing_headers = self.headers or []
        existing_headers.append(headers(cookies))
        self.headers = existing_headers

    def filename_or_extract_from_url(self) -> Optional[str]:
        if self.filename is not None:
            return self.filename
        last_segment = self.url.split('/')[-1] if '/' in self.url else ""
        potential = last_segment.split('?')[0] if '?' in last_segment else last_segment
        if not potential:
            return None
        parts = potential.split('.')
        if len(parts) > 1:
            ext = parts[-1]
            if 2 < len(ext) < 5:
                decoded = unquote(potential)
                return decoded
        return None

    def parse_filename(self):
        if self.filename is None:
            return
        resolution = RsResolution.from_filename(self.filename)
        if resolution != RsResolution.UNKNOWN:
            self.resolution = resolution
        video_format = RsVideoFormat.from_filename(self.filename)
        if video_format != RsVideoFormat.OTHER:
            self.video_format = video_format
        videocodec = RsVideoCodec.from_filename(self.filename)
        if videocodec != RsVideoCodec.UNKNOWN:
            self.videocodec = videocodec
        audio_list = RsAudio.list_from_filename(self.filename)
        if audio_list:
            self.audio = audio_list

        season_episode_re = re.compile(r'(?i)s(\d+)e(\d+)')
        match = season_episode_re.search(self.filename)
        if match:
            self.season = int(match.group(1))
            self.episode = int(match.group(2))

    def parse_subfilenames(self):
        if self.files is not None:
            for file in self.files:
                file.parse_filename()