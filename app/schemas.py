from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Shape of the JSON body the browser sends to POST /chat."""
    message: str


class ChatResponse(BaseModel):
    """Shape of the JSON body the server sends back."""
    response:  str
    intent:    str
    lang_name: str
    lang_code: str
    emotion:   Optional[str] = None   # None for non-mental-health intents
