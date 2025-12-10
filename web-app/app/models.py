from typing import List, Optional

from pydantic import BaseModel


class KeystrokeData(BaseModel):
    key: str
    code: str
    event_type: str
    timestamp: float


class SessionMetadata(BaseModel):
    correct_keys: int = 0
    incorrect_keys: int = 0
    accuracy_rate: float = 0.0
    target_text: str = ""
    total_keystrokes: int = 0


class RegisterRequest(BaseModel):
    username: str
    email: str
    keystrokes: List[KeystrokeData]
    keystrokes_2: Optional[List[KeystrokeData]] = None
    metadata: Optional[SessionMetadata] = None
    metadata_2: Optional[SessionMetadata] = None


class LoginRequest(BaseModel):
    username: str
    keystrokes: List[KeystrokeData]
    metadata: Optional[SessionMetadata] = None

