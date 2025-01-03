from __future__ import annotations
from abc import abstractmethod, ABC
from enum import Enum
from patient import PatientType
from dataclasses import dataclass, field
from typing import List, Type, Optional



class SectionType(Enum):
    OUTSIDE = "OUTSIDE"
    EMERGENCY = "EMERGENCY"
    WARD = "WARD"



class Capacity:
    def __init__(self, servers: int, queue: Optional[int] = None) -> None:
        self.servers = servers  # Number of concurrent servers
        self.queue = queue      # Maximum number of patients in queue (None for infinite)




