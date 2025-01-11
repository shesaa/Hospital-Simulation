from configure import Configure
import numpy as np
from patient import *
from base import SectionType


class Distribiutions:
    def __init__(self, cnfg: Configure = None):
        self.cnfg = cnfg
        pass
        
    def generate_next_patient_time(self, patient_type: str) -> float:
        return 1 + (hash(patient_type) % 5)
    
    def generate_next_patient_type(self):
        return np.random.choice([PatientType.ELECTIVE, PatientType.NON_ELECTIVE],1,
                                p=[0.75, 0.25])[0]
    
    def generate_service_time(self, patient: Patient) -> float:
        return 10 if patient.section == SectionType.EMERGENCY else 5


