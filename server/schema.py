from pydantic import BaseModel
from typing import Optional

class SessionCreate(BaseModel):
    name: str
    project: Optional[str] = None

class JobCreate(BaseModel):
    session_uid: str
    title: str
    poscar: str                # plain text
    ase_opt_py: str            # plain text
    ase_sh: str                # plain text

class JobId(BaseModel):
    job_uid: str