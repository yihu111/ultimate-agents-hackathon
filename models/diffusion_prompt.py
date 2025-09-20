from pydantic import BaseModel


class DiffusionPrompt(BaseModel):
    text: str


