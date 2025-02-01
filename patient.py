from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Type, Optional
from dataclasses import dataclass, field
from base import SectionType
# from events import Events


class PatientType(Enum):
    ELECTIVE = "Elective"
    NON_ELECTIVE = "NonElective"

class SurgeryType(Enum):
    SIMPLE = "Simple"
    MEDIOCRE = "Mediocre"
    COMPLICATED_HEART_DISEASE = "Complicated: Heart Disease"
    COMPLICATED_NO_HEART_DISEASE = "Complicated: No Heart Disease"

@dataclass
class Patient:
    id: int
    patient_initial_type: PatientType
    patient_type: PatientType
    surgery_type: SurgeryType
    section: SectionType
    section_entry_leave_time: dict = field(default_factory=dict)
    queue_entry_leave_time: dict = field(default_factory=dict)
    tested_at_lab: bool = False
    re_surgery_times: int = 0
    queue_entry_time: float = None
    # service_start_time: float = None
    # events_list_patient: List[Events] = []


    def to_dict(self):
        return {
            "id": self.id,
            "patient_initial_type": self.patient_initial_type.value,
            "patient_type": self.patient_type.value,
            "surgery_type": self.surgery_type.value,
            "section": self.section.value,
            # "re_surgery" : self.re_surgery
        }
    