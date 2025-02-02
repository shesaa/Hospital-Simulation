from configure import *
import numpy as np
from patient import *
from base import SectionType


# SIMULATION_SPEED = 1.0    # Multiplier for simulation speed (1.0 = real-time)
SIMULATION_CLOCK = 0.0    # Simulation time in seconds
SIMULATION_SPEED = 3600
SIMULATION_DURATION = 3600

import random
import math


# import random
# import math

# # === Module-Level Constants ===
# # Uniform distribution parameters
# GROUP_SIZE_A = 2            # Minimum group size for non‐elective entrance
# GROUP_SIZE_B = 5            # Maximum group size for non‐elective entrance
# TESTING_TIME_A = 28         # Minimum testing time
# TESTING_TIME_B = 32         # Maximum testing time
# POWER_OUTAGE_A = 1          # Minimum power outage parameter
# POWER_OUTAGE_B = 30         # Maximum power outage parameter

# # Triangular distribution parameters for EP_waiting_time_after_tests
# EP_WAIT_MIN = 5
# EP_WAIT_MODE = 75
# EP_WAIT_MAX = 100

# # Exponential distribution parameters (in minutes)
# BEDRIDDEN_TIME_ICU = 25  # For ICU/CCU
# BEDRIDDEN_TIME_GENERAL = 40  # For general ward patients

# # Normal distribution parameters for surgery times (in minutes)
# # SIMPLE surgery
# SIMPLE_MU = 27
# SIMPLE_SIGMA = 2
# # MEDIOCRE surgery
# MEDIOCRE_MU = 65
# MEDIOCRE_SIGMA = 3
# # COMPLEX surgery
# COMPLEX_MU = 215
# COMPLEX_SIGMA = 40

# # Probability thresholds (as used in generate_next_patient_type, etc.)
# ELECTIVE_PROB_THRESHOLD = 0.25
# GROUP_NON_ELECTIVE_THRESHOLD = 0.005
# # For surgery type generation (simple: 0.5, mediocre: 0.95, then complex)
# SIMPLE_THRESHOLD = 0.5
# MEDIOCRE_THRESHOLD = 0.95
# # For complex surgery re-surgery chance
# RE_SURGERY_PROB_THRESHOLD = 0.01

# # For successful surgery (10% failure)
# SUCCESSFUL_SURGERY_THRESHOLD = 0.1



# === Module-Level Constants ===
# Uniform distribution parameters
GROUP_SIZE_A = 2            # Minimum group size for non‐elective entrance
GROUP_SIZE_B = 5            # Maximum group size for non‐elective entrance
TESTING_TIME_A = 28         # Minimum testing time
TESTING_TIME_B = 32         # Maximum testing time
POWER_OUTAGE_A = 1          # Minimum power outage parameter
POWER_OUTAGE_B = 30         # Maximum power outage parameter

# Triangular distribution parameters for EP_waiting_time_after_tests
EP_WAIT_MIN = 5
EP_WAIT_MODE = 75
EP_WAIT_MAX = 100

# Exponential distribution parameters (in minutes)
BEDRIDDEN_TIME_ICU_CCU = 25  # For ICU/CCU
BEDRIDDEN_TIME_GENERAL = 50  # For general ward patients

# Normal distribution parameters for surgery times (in minutes)
# SIMPLE surgery
SIMPLE_MU = 30.222215
SIMPLE_SIGMA = 4.95765
# MEDIOCRE surgery
MEDIOCRE_MU = 74.54242222222223
MEDIOCRE_SIGMA = 9.950457667841523
# COMPLEX surgery
COMPLEX_MU = 242.0312
COMPLEX_SIGMA = 63.1162

# Probability thresholds (as used in generate_next_patient_type, etc.)
ELECTIVE_PROB_THRESHOLD = 0.25
GROUP_NON_ELECTIVE_THRESHOLD = 0.005
# For surgery type generation (simple: 0.5, mediocre: 0.95, then complex)
SIMPLE_THRESHOLD = 0.5
MEDIOCRE_THRESHOLD = 0.95
COMPLICATED_NO_HEART_DISEASE_THRESHOLD = 0.9875
# For complex surgery re-surgery chance
RE_SURGERY_PROB_THRESHOLD = 0.01

# For successful surgery (10% failure)
SUCCESSFUL_SURGERY_THRESHOLD = 0.1

# === Distribiutions Class ===

class Distribiutions:
    def __init__(self, cnfg=None, seed=350):
        self.cnfg = cnfg
        self.patient_id = 0  # Unique identifier for patients
        # Fix the random seed for reproducibility
        random.seed(seed)
        # Optionally, you could also seed numpy if used:
        # np.random.seed(seed)
    
    def uniform_dist(self, a, b):
        r = random.random()
        # Return an integer uniformly between a and b inclusive.
        return a + int(r * (b - a + 1))
    
    def triangular_dist(self, min_val, mode, max_val):
        r = random.random()
        if r < (mode - min_val) / (max_val - min_val):
            return min_val + math.sqrt((mode - min_val) * (max_val - min_val) * r)
        else:
            return max_val - math.sqrt((max_val - mode) * (max_val - min_val) * (1 - r))
    
    def exponential_dist(self, mean_minutes):
        lambda_param = 1 / mean_minutes
        r = random.random()
        time_minutes = -math.log(r) / lambda_param
        return time_minutes
    
    def normal_dist(self, mu, sigma):
        # Box-Muller transform
        r1 = random.random()
        r2 = random.random()
        z = math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)
        return mu + sigma * z
    
    def generate_next_patient_time(self, patient_type):
        # Assume PatientType is defined elsewhere (e.g., an Enum with ELECTIVE and NON_ELECTIVE)
        # For ELECTIVE, use a shorter interarrival time; for NON_ELECTIVE, longer.
        if patient_type == PatientType.ELECTIVE:
            return self.exponential_dist(60) / 60  # returns hours
        else:
            return self.exponential_dist(15) / 60  # returns hours
    
    def generate_service_time(self, patient):
        # Assume patient has attributes: section, tested_at_lab, surgery_type, etc.
        # And SectionType is an Enum.
        if patient.section in [SectionType.OUTSIDE, SectionType.RIP]:
            return 0
        elif patient.section == SectionType.OPERATING_ROOMS:
            return self._generate_operating_rooms_service_time_(patient=patient)
        elif patient.section in [SectionType.WARD, SectionType.ICU, SectionType.CCU]:
            return self._generate_bedridden_time_(patient=patient)
        elif patient.section == SectionType.EMERGENCY:
            serving_time = 1 / 6  # hour
            if patient.tested_at_lab:
                serving_time = self.triangular_dist(EP_WAIT_MIN, EP_WAIT_MODE, EP_WAIT_MAX) / 60
            return serving_time
        elif patient.section == SectionType.PRE_SURGERY:
            serving_time = 1  # hour
            if patient.tested_at_lab:
                serving_time = 1 * 24  # 2 days
            return serving_time
        elif patient.section == SectionType.LABRATORY:
            return self.uniform_dist(TESTING_TIME_A, TESTING_TIME_B) / 60
    
    def _generate_operating_rooms_service_time_(self, patient):
        if patient.surgery_type == SurgeryType.SIMPLE:
            return self.normal_dist(SIMPLE_MU, SIMPLE_SIGMA) / 60
        elif patient.surgery_type == SurgeryType.MEDIOCRE:
            return self.normal_dist(MEDIOCRE_MU, MEDIOCRE_SIGMA) / 60
        else:  # Assume COMPLEX surgery
            return self.normal_dist(COMPLEX_MU, COMPLEX_SIGMA) / 60
    
    def generate_next_patient_type(self):
        r = random.random()
        return PatientType.NON_ELECTIVE if r < ELECTIVE_PROB_THRESHOLD else PatientType.ELECTIVE
    
    def _generate_bedridden_time_(self, patient):
        if patient.section == SectionType.WARD:
            return self.exponential_dist(BEDRIDDEN_TIME_GENERAL)
        else:
            return self.exponential_dist(BEDRIDDEN_TIME_ICU_CCU)
    
    def generate_next_patient_surgery_type(self):
        r = random.random()
        if r < SIMPLE_THRESHOLD:
            return SurgeryType.SIMPLE
        elif r < MEDIOCRE_THRESHOLD:
            return SurgeryType.MEDIOCRE
        elif r < COMPLICATED_NO_HEART_DISEASE_THRESHOLD:
            return SurgeryType.COMPLICATED_NO_HEART_DISEASE
        else:
            return SurgeryType.COMPLICATED_HEART_DISEASE

    def complex_surgery_transfer_section(self, patient: Patient):
        if not self.successful_surgery():
            return SectionType.RIP
        return SectionType.ICU if patient.surgery_type== SurgeryType.COMPLICATED_NO_HEART_DISEASE else SectionType.CCU
    
    def mediocre_surgery_transfer_section(self):
        r = random.random()
        if r < 0.7:
            return SectionType.WARD
        elif r < 0.8:
            return SectionType.ICU
        else:
            return SectionType.CCU

    def successful_surgery(self):
        # for complex surgery
        # if False, RIP
        r = random.random()
        return False if r < SUCCESSFUL_SURGERY_THRESHOLD else True

    def need_for_resurgery_after_complex_surgery(self):
        r = random.random()
        return True if r < RE_SURGERY_PROB_THRESHOLD else False

    def group_non_elective_entrance(self):
        r = random.random()
        return True if r < GROUP_NON_ELECTIVE_THRESHOLD else False
    
    def generate_next_group_of_patients(self, patients_type) -> list:
        number_of_patients = 1
        patients = []
        if patients_type == PatientType.NON_ELECTIVE and self.group_non_elective_entrance():
            number_of_patients = self.uniform_dist(GROUP_SIZE_A, GROUP_SIZE_B)
        for n in range(number_of_patients):
            self.patient_id += 1
            surgery_type = self.generate_next_patient_surgery_type()
            new_patient = Patient(
                id=self.patient_id,
                patient_initial_type=patients_type,
                patient_type=patients_type,
                surgery_type=surgery_type,
                section=SectionType.OUTSIDE
            )
            patients.append(new_patient)
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
# print(a.complex_surgery_transfer_section(patient=new_patient))
# print(a.mediocre_surgery_transfer_section())
# print(a.successful_surgery())
# print(a.need_for_resurgery_after_complex_surgery())
# print(a.group_non_elective_entrance())
# print(a.exponential_dist(50))
# print(a.exponential_dist(50* 60))
# print(a.exponential_dist(50* 60) / 60)
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


