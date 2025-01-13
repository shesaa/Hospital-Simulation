from __future__ import annotations
from abc import abstractmethod, ABC
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Type, Optional



class SectionType(Enum):
    OUTSIDE = "OUTSIDE"
    PRE_SURGERY = "PRE_SURGERY"
    EMERGENCY = "EMERGENCY"
    LABRATORY = "LABRATORY"
    OPERATING_ROOMS = "OPERATING_ROOMS"
    WARD = "WARD"
    ICU = "ICU"
    CCU = "CCU"
    RIP = "RIP"



class Capacity:
    def __init__(self, servers: int, queue: Optional[int] = None) -> None:
        self.servers = servers  # Number of concurrent servers
        self.queue = queue      # Maximum number of patients in queue (None for infinite)




