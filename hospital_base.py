from __future__ import annotations
from abc import abstractmethod, ABC
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import asyncio
import time
import dash
from dash import dcc, html, Output, Input, State
import dash_canvas
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
import threading

from configure import *
from utils import *
from comminucation import *

st = time.time()


# Shared Simulation State
simulation_state = {
    "OUTSIDE": {
        "entities": [],
        "queue": []
    },
    "EMERGENCY": {
        "entities": [],
        "queue": []
    },
    "WARD": {
        "entities": [],
        "queue": []
    },

    "OPERATING_ROOMS": {
        "entities": [],
        "queue": []
    },
    "LABRATORY": {
        "entities": [],
        "queue": []
    },
    "PRE_SURGERY": {
        "entities": [],
        "queue": []
    },
    "CCU": {
        "entities": [],
        "queue": []
    },
    "ICU": {
        "entities": [],
        "queue": []
    },
    "RIP": {
        "entities": [],
        "queue": []
    },
}


# Define the Section and Hospital Classes
class Section(ABC):
    LEDGER_QUEUE = True
    __instances__: Dict[str, 'Section'] = {}
    _section_name_ = ""
    dist = Distribiutions()

    def __init__(self, section_type: SectionType, capacity: Capacity, simulation_state: Dict) -> None:
        self.section_type = section_type
        self.capacity = capacity

        self.entities: List[Patient] = []
        self.queue = asyncio.Queue(maxsize=capacity.queue) if capacity.queue is not None else asyncio.Queue()
        
        self.running = True
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Initially not paused
        self.workers: List[asyncio.Task] = []
        
        self.simulation_state = simulation_state
        Section.__instances__[self.section_type.value] = self

        self.queue_lock = asyncio.Lock()

    async def get_next_patient(self) -> Optional[Patient]:
        async with self.queue_lock:
            patients_to_process = []
            # Drain the queue
            while not self.queue.empty():
                patients_to_process.append(await self.queue.get())

            if not patients_to_process:
                return None

            # Sort patients by their priority (lowest number first: Type & entrance Time)
            patients_to_process.sort(key=lambda p: self._compute_priority_(patient=p))


            highest_priority_patient = patients_to_process.pop(0)

            # Reinsert remaining patients back into the queue
            for patient in patients_to_process:
                await self.queue.put(patient)

            return highest_priority_patient
        

    def _compute_priority_(self, patient: Patient) -> tuple:
        """
        Compute a composite priority key for sorting.
        Lower tuple values indicate higher priority.
        """

        type_priority_map = {
            'NonElective': 0,
            'Elective': 1}

        type_priority = type_priority_map.get(patient.patient_type.value)

        # If not set, default to a large number.
        entry_time = patient.queue_entry_time or float('inf')

        return (type_priority, entry_time)

    def request_handler(self, message: Request) -> Response:
        print(f"[{self.section_type.value}] Message received from {message.section_req.value} / message : {message.section_from.value} to {message.section_to.value}/ type: {message._type_} / patient id : {message.patient.id}")
        if message._type_ == RequestType.CLIENT_TO_SERVER:
            # Handling client to server requests (e.g., adding to queue)
            try:
                if message.section_req == SectionType.LABRATORY:
                    # self.move_patient(patient=message.patient, target_section=message.section_to)
                    sec_to_change = SectionType.EMERGENCY if message.patient.patient_type == PatientType.NON_ELECTIVE else SectionType.PRE_SURGERY
                    simulation_state[sec_to_change.value]["entities"].append(message.patient)
                    simulation_state[SectionType.LABRATORY.value]["entities"].remove(message.patient)
                    return Response(request=message, status=ResponseStatus.SENT)
                self.queue.put_nowait(message.patient)
                print(f"[{self.section_type.value}] Patient {message.patient.id} added to queue.")
                simulation_state[self.section_type.value]["queue"].append(message.patient)
                return Response(request=message, status=ResponseStatus.ACCEPTED)
            except asyncio.QueueFull:
                print(f"[{self.section_type.value}] Queue full. Patient {message.patient.id} rejected.")
                return Response(request=message, status=ResponseStatus.REJECTED)
        else:
            # Handling server to client requests (e.g., moving patient out)
            # print(30*"%")
            # print("Handling server to client requests (e.g., moving patient out)")
            # print(self.section_type, "section_type")
            # print(message.patient.section, "message.patient.section")
            # print(message.section_req, "section_req")
            # print(message.section_from, "section_from")
            # print(message.section_to, "section_to")
            # print(30*"%")
            if message.patient in self.entities:
                # print("here4")
                print(f"patient id {message.patient.id} to {message.section_to}")
                self.move_patient(patient=message.patient, target_section=message.section_to)
                # self.entities.remove(message.patient)
                # print(f"[{self.section_type.value}] Patient {Patient.id} removed from entities.")
                # simulation_state[self.section_type.value]["entities"].remove(message.patient)
                return Response(request=message, status=ResponseStatus.SENT)
            print(f"[{self.section_type.value}] Patient not found in entities.")
            return Response(request=message, status=ResponseStatus.REJECTED)

    async def request_sender(self, message: Request) -> Response:
        # print("here3")
        if message._type_ == RequestType.CLIENT_TO_SERVER:
            section = Section.__instances__.get(message.section_to.value)
        else:
            section = Section.__instances__.get(message.section_from.value)
        # print(section.value)
        if not section:
            print(f"Section {message.section_to.value} does not exist.")
            return Response(request=message, status=ResponseStatus.REJECTED)
        # print("here")
        response = section.request_handler(message)
        # Update simulation_state
        if response.status == ResponseStatus.ACCEPTED:
            # self.simulation_state[self.section_type.value]["queue"].append(message.patient)
            pass
        elif response.status == ResponseStatus.REJECTED:
            pass  # Handle rejection if needed
        elif response.status == ResponseStatus.SENT:
            # Move patient to target section
            target_section = Section.__instances__.get(message.section_to.value)
            if target_section:
                # message.patient.section = target_section.section_type
                # target_section.entities.append(message.patient)
                # self.simulation_state[self.section_type.value]["entities"].remove(message.patient)
                # self.simulation_state[message.section_to.value]["entities"].append(message.patient)
                # self.simulation_state[self.section_type.value]["entities"].append(message.patient)
                # self.move_patient(patient=message.patient, target_section=target_section)
                pass
        return response

    async def run(self, distrib: Distribiutions):
        print(f"[{self._section_name_}] Section started.")
        self.start_workers(distrib)
        while self.running:
            await self.pause_event.wait()
            await asyncio.sleep(0.1 / SIMULATION_SPEED)  # Idle loop; actual processing is done by workers

    async def stop(self):
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        print(f"[{self._section_name_}] Section stopped.")

    async def pause(self):
        self.pause_event.clear()
        print(f"[{self.section_type.value}] Paused.")

    async def resume(self):
        self.pause_event.set()
        print(f"[{self.section_type.value}] Resumed.")

    @abstractmethod
    def decide_next_section(self, patient: Patient) -> SectionType:
        return
    
    @abstractmethod
    async def section_process(self):
        return
    

    async def worker_task(self, worker_id: int, distrib: Distribiutions):
        '''
        Number of workers depends on the capacity of a section
        serving many patient simultaneously

        Also client_to_server requests are made here
        
        '''
        print(f"[{self.section_type.value}] Worker {worker_id} started.")
        
        while self.running:
            await self.pause_event.wait()  # Wait if paused
            try:
                # patient = await self.queue.get()
                patient = await self.get_next_patient()
                if patient is None:
                    # No patient to process, wait briefly
                    await asyncio.sleep(0.1 / SIMULATION_SPEED)
                    continue

                print("----------"* 6 , "Block")
                
                move_request = Request(
                        section_req= self.section_type,
                        section_from=patient.section,
                        section_to=self.section_type,
                        patient=patient,
                        _type_=RequestType.SERVER_TO_CLIENT
                    )
                
                response = await self.request_sender(move_request)
                print("block_time:", (time.time()- st)/ SIMULATION_SPEED, "hours")

                print(f"patient {patient.id} 's moved from queue into the {self.section_type.value} entities")
                print(f"[{self.section_type.value}] Worker {worker_id} serving patient: {patient.id}")

                # Simulate serving time
                serve_time = distrib.generate_service_time(patient)
                print(f"serve_time for patient {patient.id}: {serve_time}")

                s = time.time()
                patient.service_start_time = (s - st) * 3600 / SIMULATION_SPEED

                await asyncio.shield(asyncio.sleep(serve_time * 3600 / SIMULATION_SPEED))
                print(f"[{self.section_type.value}] Worker {worker_id} finished serving patient: {patient.id}")
                print("duration :", (time.time() - s)* 3600/ SIMULATION_SPEED, "hours")

                if self.section_type == SectionType.LABRATORY:
                    patient.tested_at_lab = True
                
                # After serving, time to take a step for entering next section
                section_to = self.decide_next_section(patient=patient)
                move_request = Request(
                    section_req= self.section_type,
                    section_from=self.section_type,
                    section_to=section_to,
                    patient=patient,
                    _type_=RequestType.CLIENT_TO_SERVER
                )
                response = await self.request_sender(move_request)
                is_moved = await self.wait_for_section(patient=patient, specific_section=section_to)
                if is_moved:
                    print(f"[{self.section_type.value}] Worker {worker_id} moved patient {patient.id} to Ward.")
                
                if section_to== SectionType.LABRATORY:
                    print(f"[{self.section_type.value}] Worker {worker_id} waiting for patient {patient.id} to come back from {section_to.value}.")
                    is_backed = await self.wait_for_section(patient=patient, specific_section=self.section_type)
                    print(f"[{self.section_type.value}] patient {patient.id} is now backed from {section_to}!.")
                # elif self.section_type == SectionType.WARD:
                #     # Handle discharge or other logic
                #     print(f"[{self.section_type.value}] Patient {patient.id} discharged.")
                #     # Remove patient from ward
                #     self.entities.remove(patient)
                #     simulation_state[self.section_type.value]["entities"].remove(patient)
                #     patient.section = SectionType.OUTSIDE
                    
                if self.section_type == SectionType.OPERATING_ROOMS:
                    print(f"[{self.section_type.value}] Worker {worker_id} is preparing surgery room for next patient...")
                    await asyncio.shield(asyncio.sleep((1/6) * 3600 / SIMULATION_SPEED))
                    print(f"[{self.section_type.value}] Worker {worker_id} preparing surgery room DONE!")

                self.queue.task_done()
            except asyncio.CancelledError:
                print(f"[{self.section_type.value}] Worker {worker_id} cancelled.")
                break
            except Exception as e:
                print(f"[{self.section_type.value}] Worker {worker_id} encountered an error: {e}")
        
    def start_workers(self, distrib: Distribiutions):
        for i in range(self.capacity.servers):
            worker_task = asyncio.create_task(self.worker_task(worker_id=i+1, distrib=distrib))
            self.workers.append(worker_task)

    async def stop_workers(self):
        self.running = False
        for worker_task in self.workers:
            worker_task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        print(f"[{self.section_type.value}] All workers stopped.")


    def move_patient(self, patient: Patient, target_section: SectionType):
        '''
        Moves a patient to the target section.
        '''
        # print("state BEFORE update")
        # print(simulation_state)
        # Update patient's section

        self.entities.remove(patient)
        target_section_instance = Section.__instances__.get(target_section.value)

        condition1 = not target_section_instance == SectionType.LABRATORY
        condition2 = not (target_section_instance in [SectionType.EMERGENCY, SectionType.PRE_SURGERY] and  patient.tested_at_lab)
        
        if condition1:
            simulation_state[self.section_type.value]["entities"].remove(patient)
        if condition2:
            simulation_state[target_section.value]["queue"].remove(patient)
            
        patient.section = target_section

        target_section_instance.entities.append(patient)
        
        if condition2:
            simulation_state[target_section.value]["entities"].append(patient)
        # print("hahaha")

        print(f"[{self.section_type.value}] Patient {patient.id} moved to {target_section.value}")
        # print("state AFTER update")
        # print(simulation_state)

    async def wait_for_section(self, patient: Patient, specific_section: SectionType, timeout: float = 100):
        '''
        Waits until the patient's section matches the specific section or until timeout.
        '''
        # print("error")
        start_time = time.time()
        # print("error2")
        while patient.section != specific_section:
            if (time.time() - start_time) > timeout:
                print(f"Timeout: Patient {patient.id} did not move to {specific_section.value} within {timeout} seconds.")
                return False
            await asyncio.sleep(0.1 / SIMULATION_SPEED)  # Wait for 100ms before checking again
        return True
    

    async def reduce_capacity(self, reduction_percent: float):
        """
        Reduces the number of active workers by a given percentage.
        """
        original_servers = self.capacity.servers
        # Calculate new server count after reduction
        new_server_count = max(1, int(original_servers * (1 - reduction_percent)))

        print(f"[{self.section_type.value}] Reducing capacity from {original_servers} to {new_server_count} servers.")

        # Cancel excess workers
        while len(self.workers) > new_server_count:
            worker = self.workers.pop()
            worker.cancel()
            try:
                await worker  # Awaiting ensures the worker finishes its current task due to shield
            except asyncio.CancelledError:
                print(f"[{self.section_type.value}] Cancelled a worker due to capacity reduction.")

        # Update capacity attribute
        self.capacity.servers = new_server_count


    async def restore_capacity(self, target_servers: int, distrib: Distribiutions):
        """
        Restores capacity by spawning additional workers until reaching target_servers.
        """
        current_servers = self.capacity.servers
        print(f"[{self.section_type.value}] Restoring capacity from {current_servers} to {target_servers} servers.")

        # Spawn new workers if needed
        for _ in range(target_servers - current_servers):
            worker_id = len(self.workers) + 1
            new_worker = asyncio.create_task(self.worker_task(worker_id, distrib))
            self.workers.append(new_worker)

        # Update capacity attribute
        self.capacity.servers = target_servers

class Hospital:
    class Emergency(Section):
        _section_name_ = SectionType.EMERGENCY.value
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.EMERGENCY, capacity, simulation_state)
        
        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.LABRATORY
            if patient.tested_at_lab:
                next_section_type = SectionType.OPERATING_ROOMS
            return next_section_type

        async def section_process(self):
            # Placeholder loop: perform emergency-specific tasks
            while self.running:
                # Insert emergency-specific processing here
                await asyncio.sleep(1)  # Simulate periodic work

    class Ward(Section):
        _section_name_ = SectionType.WARD.value
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.WARD, capacity, simulation_state)
        
        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.OUTSIDE
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert ward-specific processing here
                await asyncio.sleep(1)

    class CCU(Section):
        _section_name_ = SectionType.CCU.value
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.CCU, capacity, simulation_state)

        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.OUTSIDE
            if self.dist.need_for_resurgery_after_complex_surgery():
                next_section_type = SectionType.OPERATING_ROOMS    
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert CCU-specific processing here
                await asyncio.sleep(1)

    class ICU(Section):
        _section_name_ = SectionType.ICU.value
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.ICU, capacity, simulation_state)

        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.OUTSIDE
            if self.dist.need_for_resurgery_after_complex_surgery():
                next_section_type = SectionType.OPERATING_ROOMS
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert ICU-specific processing here
                await asyncio.sleep(1)

    class Labratory(Section):
        _section_name_ = SectionType.LABRATORY.value
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.LABRATORY, capacity, simulation_state)

        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.PRE_SURGERY
            if patient.patient_type == PatientType.NON_ELECTIVE:
                next_section_type = SectionType.EMERGENCY
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert laboratory-specific processing here
                await asyncio.sleep(1)

    class PreSurgery(Section):
        _section_name_ = SectionType.PRE_SURGERY.value
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.PRE_SURGERY, capacity, simulation_state)

        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.LABRATORY
            if patient.tested_at_lab:
                next_section_type = SectionType.OPERATING_ROOMS
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert pre-surgery-specific processing here
                await asyncio.sleep(1)

    class OperatingRooms(Section):
        _section_name_ = SectionType.OPERATING_ROOMS.value
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.OPERATING_ROOMS, capacity, simulation_state)

        def decide_next_section(self, patient: Patient):
            if patient.surgery_type == SurgeryType.SIMPLE:
                next_section_type = SectionType.WARD
            elif patient.surgery_type == SurgeryType.MEDIOCRE:
                next_section_type = self.dist.mediocre_surgery_transfer_section()
            else: # COMPLEX
                next_section_type = self.dist.complex_surgery_transfer_section()
                if not self.dist.successful_surgery():
                    # DEATH : RIP
                    next_section_type = SectionType.RIP
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert operating room-specific processing here
                await asyncio.sleep(1)
            

    def __init__(self, simulation_state: Dict):
        self.emergency = self.Emergency(SECTION_CAPACITIES.get(SectionType.EMERGENCY), simulation_state)
        self.ward = self.Ward(SECTION_CAPACITIES.get(SectionType.WARD), simulation_state)
        self.ccu = self.CCU(SECTION_CAPACITIES.get(SectionType.CCU), simulation_state)
        self.icu = self.ICU(SECTION_CAPACITIES.get(SectionType.ICU), simulation_state)
        self.operating_rooms = self.OperatingRooms(SECTION_CAPACITIES.get(SectionType.OPERATING_ROOMS), simulation_state)
        self.labratory = self.Labratory(SECTION_CAPACITIES.get(SectionType.LABRATORY), simulation_state)
        self.pre_surgery = self.PreSurgery(SECTION_CAPACITIES.get(SectionType.PRE_SURGERY), simulation_state)

        self.sections = {
            SectionType.EMERGENCY: self.emergency,
            SectionType.WARD: self.ward,
            SectionType.CCU: self.ccu,
            SectionType.ICU: self.icu,
            SectionType.OPERATING_ROOMS: self.operating_rooms,
            SectionType.LABRATORY: self.labratory,
            SectionType.PRE_SURGERY: self.pre_surgery,
        }

    def get_section(self, section_type: SectionType) -> Optional[Section]:
        return self.sections.get(section_type)
    
    async def electricity_suspension(self):
        # ICU and CCU
        self.icu.reduce_capacity(reduction_percent=REDUCTION_PERCENT)
        self.ccu.reduce_capacity(reduction_percent=REDUCTION_PERCENT)
        await asyncio.sleep(electricity_suspension_hours * 3600 / SIMULATION_SPEED) # 24 hours
        

class ClientGeneratorForHospital(Section):
    def __init__(self, targeted_hospital: Hospital, dist: Distribiutions, simulation_state: Dict):
        self.section_type = SectionType.OUTSIDE
        self.entities: List[Patient] = []
        self.targeted_hospital = targeted_hospital
        self.dist = dist
        self.running = True
        self.simulation_state = simulation_state
        Section.__instances__[self.section_type.value] = self

    async def run(self):
        print("[ClientGenerator] Started.")
        while self.running:
            print("1")
            next_patients_type = self.dist.generate_next_patient_type()
            print("2", next_patients_type)
            next_patient_interval = self.dist.generate_next_patient_time(patient_type=next_patients_type)
            print("3",next_patient_interval)
            list_of_patients = self.dist.generate_next_group_of_patients(next_patients_type)
            print("4", list_of_patients)
            print(f"[ClientGenerator] New patients in {next_patient_interval}")
            
            await asyncio.sleep(next_patient_interval * 3600 / SIMULATION_SPEED)  # Simulate patient arrival interval

            for p in list_of_patients:
                self.entities.append(p)
                simulation_state["OUTSIDE"]["entities"].append(p)
                print(f"[ClientGenerator] Generated new patient: {p.id} (Type: {next_patients_type.value})")
                # Create and send request
                section_to = SectionType.EMERGENCY if p.patient_type == PatientType.NON_ELECTIVE else SectionType.PRE_SURGERY
                
                request = Request(
                    section_req = self.section_type,
                    section_from=SectionType.OUTSIDE,
                    section_to=section_to,
                    patient=p,
                    _type_=RequestType.CLIENT_TO_SERVER
                )
                response = self.targeted_hospital.emergency.request_handler(message=request)
            # response = await self.request_sender(message=request)
            # if response.status == ResponseStatus.ACCEPTED:
            #     simulation_state["EMERGENCY"]["queue"].append(new_patient)
                print(f"[ClientGenerator] Sent entry request for patient: {p.id} - Response: {response.status.value}")

    async def stop(self):
        self.running = False
        print("[ClientGenerator] Stopping...")
        print("[ClientGenerator] Stopped.")

    def decide_next_section(self, patient):
        return super().decide_next_section(patient)
    
    async def section_process(self):
        return await super().section_process()

class Nature:
    # class RIP(Section):
        # pass

    def __init__(self, targeted_hospital: Hospital, dist: Distribiutions, simulation_state: Dict):
        self.running = True


    async def run(self):
        print("[Nature] Started.")
        while self.running:
            pass