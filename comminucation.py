
from __future__ import annotations
from abc import abstractmethod, ABC
from enum import Enum
from patient import PatientType
from dataclasses import dataclass, field
from typing import List, Type, Optional
from base import *
from patient import *


class RequestType(Enum):
    SERVER_TO_CLIENT = "Move / Change Section / Out Of the Queue"
    CLIENT_TO_SERVER = "Checking capacity"

@dataclass
class Request:
    section_from: SectionType
    section_to: SectionType
    patient: Patient
    _type_: RequestType

class ResponseStatus(Enum):
    ACCEPTED = "Accepted"  # accepted to the queue
    REJECTED = "Rejected"
    WAIT = "Wait"
    SENT = "Patient has begun to move between section"

@dataclass
class Response:
    request: Request
    status: ResponseStatus