from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Type, Optional
from dataclasses import dataclass, field
import time
from patient import Patient, PatientType

class EventType(Enum):
    PA = "Patient Arrival"
    AW = "Administrative Works"
    LT = "Labratory Arrival"
    SRT = "Surgery Room"
    GT = "Ward"
    ICUT = "ICU"
    CCUT = "CCU"
    PC_PD = "Power Connected / Power Disconnected"
    NRS = "Need for Re-Surgery"
    END = "End"


class Event:
    def __init__(self, _type_: EventType, _time_, patient: Optional[Patient]= None ):
        self.event_type = _type_
        self.event_time = _time_
        self.event_patient = patient
        pass