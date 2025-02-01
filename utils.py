from configure import *
import numpy as np
from patient import *
from base import SectionType


# SIMULATION_SPEED = 1.0    # Multiplier for simulation speed (1.0 = real-time)
SIMULATION_CLOCK = 0.0    # Simulation time in seconds
SIMULATION_SPEED = 3600
SIMULATION_DURATION = 3600

from configure import *
import numpy as np
import random
import math


class Distribiutions:
    def __init__(self, cnfg = None):
        self.cnfg = cnfg
        self.patient_id = 0  # Unique identifier for patients
        pass
    
    def uniform_dist(self, a, b):
        r = random.random()
        return a + int(r * (b - a + 1))   
    #for groupe size a=2, b=5 
    #for testing time a=28, b=32
    #for power outage a=1, b=30

    def triangular_dist(self, min_val, mode, max_val):
        r = random.random()
        if r < (mode - min_val) / (max_val - min_val):
            return min_val + ((mode - min_val) * (max_val - min_val) * r) ** 0.5
        else:
            return max_val - ((max_val - mode) * (max_val - min_val) * (1 - r)) ** 0.5
    #for EP_waiting_time_after_tests min_val=5, mode=75, max_val=100
    
    def exponential_dist(self, mean_minutes):
        lambda_param = 1 / mean_minutes
        r = random.random()
        time_minutes = -math.log(r) / lambda_param
        return time_minutes
    #parameter's type is minutes
    #for bedridden time in ICU/CCU parameter=25*60
    #for bedridden time in general parameter=50*60
    
    def normal_dist(self, mu, sigma):
        r1 = random.random()
        r2 = random.random()
        # Box-Muller transform to get standard normal
        z = math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)
        # Transform standard normal to desired normal
        # X = μ + σZ
        value = mu + sigma * z
        return value
    #for complex surgery mu=242.0312, sigma=63.1162
    #for moderate surgery mu=74.54242222222223, sigma=9.950457667841523
    #for simple surgery mu=30.222215, sigma=4.95765
    
    def generate_next_patient_time(self, patient_type: PatientType) -> float:
        return self.exponential_dist(15)/60 if patient_type==PatientType.ELECTIVE else self.exponential_dist(60)/60
    
    def generate_service_time(self, patient: Patient) -> float:
        if patient.section in [SectionType.OUTSIDE , SectionType.RIP]:
            return 0
        elif patient.section == SectionType.OPERATING_ROOMS:
            return self._generate_operating_rooms_service_time_(patient=patient)
        elif patient.section in [SectionType.WARD , SectionType.ICU, SectionType.CCU]:
            return self._generate_bedridden_time_(patient=patient)
        elif patient.section == SectionType.EMERGENCY:
            serving_time = 1/6 # hour
            if patient.tested_at_lab:
                serving_time = self.triangular_dist(5, 75, 100)/60
            return serving_time
        elif patient.section == SectionType.PRE_SURGERY:
            serving_time = 1 # hour
            if patient.tested_at_lab:
                serving_time = 2*24 # 2 days
            return serving_time
        elif patient.section == SectionType.LABRATORY:
            return self.uniform_dist(28, 32) / 60
        
    def _generate_operating_rooms_service_time_(self, patient: Patient):
        if patient.surgery_type == SurgeryType.SIMPLE:
            return self.normal_dist(30.222215, 4.95765) / 60
        elif patient.surgery_type == SurgeryType.MEDIOCRE:
            return self.normal_dist(74.54242222222223, 9.950457667841523) /60
        else:  # Assuming this is for SurgeryType.COMPLEX
            return self.normal_dist(242.0312, 63.1162) /60
    
    def generate_next_patient_type(self):
        r = random.random()
        return PatientType.ELECTIVE if r < 0.25 else PatientType.NON_ELECTIVE
    
    def _generate_bedridden_time_(self, patient: Patient):
        if patient.section==SectionType.WARD:
            return self.exponential_dist(50)
        else:
            return self.exponential_dist(25)
    
    def generate_next_patient_surgery_type(self):
        r = random.random()
        if r < 0.5:
            return SurgeryType.SIMPLE
        if r < 0.95:
            return SurgeryType.MEDIOCRE
        return SurgeryType.COMPLICATED_HEART_DISEASE

    def complex_surgery_transfer_section(self):
        r = random.random()
        return SectionType.ICU if r < 0.75 else SectionType.CCU
    
    def mediocre_surgery_transfer_section(self):
        r = random.random()
        return SectionType.WARD if r < 0.7 else SectionType.ICU if r < 0.8 else SectionType.CCU

    def successful_surgery(self):
        r = random.random()
        return False if r < 0.1 else True

    def need_for_resurgery_after_complex_surgery(self):
        r = random.random()
        return True if r < 0.01 else False


    def group_non_elective_entrance(self):
        r = random.random()
        return True if r < 0.005 else False
    
    def generate_next_group_of_patients(self, patients_type: PatientType) -> list:
        number_of_patients = 1
        patients = []
        if patients_type == PatientType.NON_ELECTIVE and self.group_non_elective_entrance():
            number_of_patients = self.uniform_dist(2, 5)
        # print("p", number_of_patients)
        for n in range(number_of_patients):
            self.patient_id += 1
            # print("b", n)
            surgery_type = self.generate_next_patient_surgery_type()
            print(surgery_type)
            new_patient = Patient(
                id=self.patient_id,
                patient_initial_type=patients_type,
                patient_type=patients_type,
                surgery_type=surgery_type,
                section= SectionType.OUTSIDE
            )
            print("a",n)

            patients.append(new_patient)
        # print("huh")
        return patients




# new_patient = Patient(
#     id=1,
#     patient_initial_type=PatientType.ELECTIVE,
#     patient_type=PatientType.ELECTIVE,
#     surgery_type=SurgeryType.MEDIOCRE,
#     section= SectionType.OUTSIDE
# )

# a = Distribiutions()
# print(a.generate_next_patient_time(patient_type=new_patient.patient_type))
# print(a.generate_next_patient_type())
# print(20* '*')
# for i in SectionType:
#     new_patient.section = i
#     print(i)
#     print(a.generate_service_time(new_patient))
#     print(20* '*')

# print(a.generate_next_patient_surgery_type())
# print(a.complex_surgery_transfer_section())
# print(a.mediocre_surgery_transfer_section())
# print(a.successful_surgery())
# print(a.need_for_resurgery_after_complex_surgery())
# print(a.group_non_elective_entrance())
from typing import List, Type, Optional, Tuple


def n_nonelective_n_elective(patient_list: List[Patient]) -> tuple:
    n_nonelective = 0
    n_elective = 0
    print("hoy0")
    print(patient_list)
    for p in patient_list:
        print("hoy")
        if p.patient_type == PatientType.NON_ELECTIVE:
            n_nonelective += 1
            continue
        n_elective += 1
    return n_nonelective , n_elective


# phase 2:
# patient start time and end time for each section
# capturing emergency queue by time: average
# max & mean of queue length & waiting time of all sections
# how many times re-surgery : average
# average of efficiency of beds of each section

# patient class: number of re-surgery /// start and end time of each section


