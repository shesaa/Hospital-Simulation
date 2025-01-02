from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Type, Optional
from dataclasses import dataclass, field


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
    patient_type: PatientType
    surgery_type: SurgeryType

    def to_dict(self):
        return {
            "id": self.id,
            "patient_type": self.patient_type,
            "surgery_type": self.surgery_type.value
        }
    