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

from events import *

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
    all_events = []
    all_patients = []

    def __init__(self, section_type: SectionType, capacity: Capacity) -> None:
        self.section_type = section_type
        self.capacity = capacity

        self.entities: List[Patient] = []
        self.queue = asyncio.Queue(maxsize=capacity.queue) if capacity.queue is not None else asyncio.Queue()
        
        self.running = True
        self.paused = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Initially not paused
        self.workers: List[asyncio.Task] = []
        self.worker_to_patient = {}
        
        Section.__instances__[self.section_type.value] = self

        self.queue_lock = asyncio.Lock()
        self.patients_to_process = []

        self.queue_size_time_series = ([], [])
        self.entity_size_time_series = ([], [])
        self.duration_serve = []


    async def get_next_patient(self) -> Optional[Patient]:
        async with self.queue_lock:
            self.patients_to_process = []
            # Drain the queue
            while not self.queue.empty():
                self.patients_to_process.append(await self.queue.get())

            if not self.patients_to_process:
                return None

            # Sort patients by their priority (lowest number first: Type & entrance Time)
            self.patients_to_process.sort(key=lambda p: self._compute_priority_(patient=p))


            highest_priority_patient = self.patients_to_process.pop(0)

            # Reinsert remaining patients back into the queue
            for patient in self.patients_to_process:
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
    
    def _determine_event_type_(self, message: Request) -> EventType:
        section_to = message.section_to
        _map_dict_ = {
            SectionType.EMERGENCY: EventType.PA,
            SectionType.PRE_SURGERY: EventType.PA,
            
            SectionType.LABRATORY: EventType.LT,
            SectionType.OPERATING_ROOMS: EventType.SRT,
            SectionType.ICU: EventType.ICUT,
            SectionType.CCU: EventType.CCUT,
            SectionType.WARD: EventType.GT,
            SectionType.OUTSIDE: EventType.END
        }
        event_type  = _map_dict_[section_to]
        if message.section_from in [SectionType.CCU, SectionType.ICU] and section_to == SectionType.OPERATING_ROOMS:
            event_type = EventType.NRS # need for re-surgery
        return event_type
    

    async def request_handler(self, message: Request) -> Response:
        print(30* "--")
        print(f"[{self.section_type.value}] Message received from {message.section_req.value} / message : {message.section_from.value} to {message.section_to.value}/ type: {message._type_} / patient id : {message.patient.id}")
        if message._type_ == RequestType.CLIENT_TO_SERVER:
            # Handling client to server requests (e.g., adding to queue)
            try:
                if message.section_req == SectionType.LABRATORY:
                    new_message = Request(self.section_type, message.section_from, message.section_to, message.patient, _type_ = RequestType.SERVER_TO_CLIENT)
                    response = await self.request_sender(message=new_message)
                    return response
                self.queue.put_nowait(message.patient)
                print(f"[{self.section_type.value}] Patient {message.patient.id} added to queue.")
                queue_entry_time = (time.time() - st) * 3600 / SIMULATION_SPEED
                message.patient.queue_entry_time = queue_entry_time
                message.patient.queue_entry_leave_time[self.section_type] = [queue_entry_time, None]
                self.queue_size_time_series[0].append((time.time() - st)* 3600/SIMULATION_SPEED)
                self.queue_size_time_series[1].append(self.queue.qsize())
                return Response(request=message, status=ResponseStatus.ACCEPTED)
            except asyncio.QueueFull:
                print(f"[{self.section_type.value}] Queue full. Patient {message.patient.id} rejected.")
                return Response(request=message, status=ResponseStatus.REJECTED)
        else:
            # Handling server to client requests (e.g., moving patient out)
            if message.patient in self.entities:
                print(f"patient id {message.patient.id} to {message.section_to}")
                self.move_patient(patient=message.patient, target_section=message.section_to)
                return Response(request=message, status=ResponseStatus.SENT)
            print(f"[{self.section_type.value}] Patient not found in entities.")
            return Response(request=message, status=ResponseStatus.REJECTED)

    async def request_sender(self, message: Request) -> Response:
        if message.section_req == SectionType.EMERGENCY and message.section_from== SectionType.LABRATORY:
            print("hah3")
        if message.section_to in [SectionType.OUTSIDE, SectionType.RIP]:
            event = Event(_type_= EventType.END,
                    _time_= (time.time() - st)* 3600/SIMULATION_SPEED,
                    patient=message.patient)
            self.all_events.append(event)
            self.move_patient(patient=message.patient, target_section=message.section_to)
            return Response(request=message, status=ResponseStatus.SENT) 
        if message._type_ == RequestType.CLIENT_TO_SERVER:
            section = Section.__instances__.get(message.section_to.value)
            if not (section.section_type in [SectionType.EMERGENCY, SectionType.PRE_SURGERY] and message.patient.tested_at_lab):
                event = Event(_type_= self._determine_event_type_(message=message),
                    _time_= (time.time() - st)* 3600/SIMULATION_SPEED,
                    patient=message.patient)
                self.all_events.append(event)
        else:
            section = Section.__instances__.get(message.section_from.value)
            if self.section_type in [SectionType.EMERGENCY, SectionType.PRE_SURGERY] and message.patient.tested_at_lab is False:
                event = Event(_type_= EventType.AW,
                    _time_= (time.time() - st)* 3600/SIMULATION_SPEED,
                    patient=message.patient)
                self.all_events.append(event)
        if not section:
            print(f"Section {message.section_to.value} does not exist.")
            return Response(request=message, status=ResponseStatus.REJECTED)
        response = await section.request_handler(message)
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
                # print(f"[{self.section_type.value}] Worker {worker_id} looking for new patient.")
                patient = await self.get_next_patient()
                if patient is None:
                    # No patient to process, wait briefly
                    await asyncio.sleep(0.2 / SIMULATION_SPEED)
                    continue

                # print("----------"* 6 , "Block")

                move_request = Request(
                        section_req= self.section_type,
                        section_from=patient.section,
                        section_to=self.section_type,
                        patient=patient,
                        _type_=RequestType.SERVER_TO_CLIENT
                    )
                print(f"[{self.section_type.value}] Worker {worker_id} requesting patient {patient.id} locating in {patient.section.value}.")
                response = await self.request_sender(move_request)
                # print("block_time:", (time.time()- st)/ SIMULATION_SPEED, "hours")

                print(f"[{self.section_type.value}] patient {patient.id} 's moved from queue into the {self.section_type.value} entities")
                time_entry = (time.time() - st) * 3600 / SIMULATION_SPEED
                patient.queue_entry_leave_time[self.section_type][1] = time_entry
                patient.section_entry_leave_time[self.section_type] = [time_entry, None, None, None]
                self.queue_size_time_series[0].append((time.time() - st)* 3600/SIMULATION_SPEED)
                self.queue_size_time_series[1].append(self.queue.qsize())
                
                # patient_start_time = (time.time() - st) * 3600 / SIMULATION_SPEED
                print(f"[{self.section_type.value}] Worker {worker_id} serving patient: {patient.id}")
                self.worker_to_patient[worker_id] = patient
                # print(self.worker_to_patient)

                # Simulate serving time
                serve_time = distrib.generate_service_time(patient)

                s = time.time()
                # patient.service_start_time = (s - st) * 3600 / SIMULATION_SPEED

                await asyncio.shield(asyncio.sleep(serve_time * 3600 / SIMULATION_SPEED))
                if self.section_type in [SectionType.EMERGENCY, SectionType.PRE_SURGERY] and patient.tested_at_lab is False:
                    print(f"[{self.section_type.value}] serving_time of administrative work for patient {patient.id}: {serve_time}")
                    print(f"[{self.section_type.value}] Worker {worker_id} finished administrative work patient: {patient.id} in", (time.time() - s)* 3600/ SIMULATION_SPEED, "hours")
                else:
                    print(f"[{self.section_type.value}] serving_time for patient {patient.id}: {serve_time}")
                    print(f"[{self.section_type.value}] Worker {worker_id} finished serving patient: {patient.id} in", (time.time() - s)* 3600/ SIMULATION_SPEED, "hours")
                patient_end_time = (time.time() - st) * 3600 / SIMULATION_SPEED
                patient.section_entry_leave_time[self.section_type][1] = patient_end_time

                if self.section_type == SectionType.LABRATORY:
                    patient.tested_at_lab = True
                
                if self.section_type in [SectionType.OUTSIDE, SectionType.RIP]:
                    self.queue.task_done()
                    continue
                
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
                    print(f"[{self.section_type.value}] Worker {worker_id} moved patient {patient.id} to {section_to.value}.")
                
                if section_to== SectionType.LABRATORY:
                    print(f"[{self.section_type.value}] Worker {worker_id} waiting for patient {patient.id} to come back from {section_to.value}.")
                    is_backed = await self.wait_for_section(patient=patient, specific_section=self.section_type)
                    print(f"[{self.section_type.value}] patient {patient.id} is now backed from {section_to}!.")
                    serve_time = distrib.generate_service_time(patient)
                    print(f"[{self.section_type.value}] remaining serving_time for patient {patient.id}: {serve_time}")
                    s = time.time()
                    time_entry = (time.time() - st) * 3600 / SIMULATION_SPEED
                    patient.section_entry_leave_time[self.section_type][2] = time_entry
                    await asyncio.shield(asyncio.sleep(serve_time * 3600 / SIMULATION_SPEED))
                    print(f"[{self.section_type.value}] Worker {worker_id} finished serving patient: {patient.id} in", (time.time() - s)* 3600/ SIMULATION_SPEED, "hours")

                    patient_end_time = (time.time() - st) * 3600 / SIMULATION_SPEED
                    patient.section_entry_leave_time[self.section_type][3] = patient_end_time

                    section_to = self.decide_next_section(patient=patient)
                    print(f"[{self.section_type.value}] patient {patient.id} now must go to {section_to.value}")
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
                        print(f"[{self.section_type.value}] Worker {worker_id} moved patient {patient.id} to {section_to.value}.")
                    
                print(f"[{self.section_type.value}] Worker {worker_id} task done (serving patient {patient.id})")

                self.worker_to_patient[worker_id] = None
                # duration_patient = patient_end_time - patient_start_time
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
        patient : from A to B
        '''

        # B instance
        target_section_instance = Section.__instances__.get(target_section.value)

        # print(len(self.entities), self.section_type)
        # print(len(target_section_instance.entities), target_section_instance.section_type)
        

        # if B is labratory
        condition1 = target_section_instance.section_type == SectionType.LABRATORY
        # if A is labratory
        condition2 = (target_section_instance.section_type in [SectionType.EMERGENCY, SectionType.PRE_SURGERY] and  patient.tested_at_lab)
        if condition2 and not self.section_type== SectionType.LABRATORY:
            print("there is a logical bug here")

        # Firstly
        if not condition2:
            target_section_instance.entities.append(patient)
        patient.section = target_section

        # Secondly
        if not condition1:
            self.entities.remove(patient)

        self.entity_size_time_series[0].append((time.time() - st)* 3600/SIMULATION_SPEED)
        self.entity_size_time_series[1].append(len(self.entities))
        target_section_instance.entity_size_time_series[0].append((time.time() - st)* 3600/SIMULATION_SPEED)
        target_section_instance.entity_size_time_series[1].append(len(target_section_instance.entities))
        # Finally
        # deleting patient from B queue

        # if condition1: #ok
        #     self.entities.remove(patient)
        #     # simulation_state[self.section_type.value]["entities"].remove(patient)
        # if condition2:
        #     # simulation_state[target_section.value]["queue"].remove(patient)
            
        
        # if condition2:
        #     target_section_instance.entities.append(patient)
        #     # simulation_state[target_section.value]["entities"].append(patient)
        # print(len(self.entities), self.section_type)
        # print(len(target_section_instance.entities), target_section_instance.section_type)

        print(f"[{self.section_type.value}] Patient {patient.id} moved to {target_section.value}")


    async def wait_for_section(self, patient: Patient, specific_section: SectionType, timeout: float = 100):
        '''
        Waits until the patient's section matches the specific section or until timeout.
        '''
        start_time = time.time()
        if specific_section == self.section_type:
            print(f"[{self.section_type.value}] Waiting to patient {patient.id} come back from {patient.section.value}")
        while patient.section != specific_section:
            if (time.time() - start_time) > timeout * SIMULATION_DURATION:
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


    async def restore_capacity(self, reduction_percent: float):
        """
        Restores capacity by spawning additional workers until reaching target_servers.
        """
        
        current_servers = self.capacity.servers
        target_servers = int(current_servers / (1-reduction_percent))
        print(f"[{self.section_type.value}] Restoring capacity from {current_servers} to {target_servers} servers.")

        # Spawn new workers if needed
        for _ in range(target_servers - current_servers):
            worker_id = len(self.workers) + 1
            new_worker = asyncio.create_task(self.worker_task(worker_id, self.dist))
            self.workers.append(new_worker)

        # Update capacity attribute
        self.capacity.servers = target_servers

class Hospital:
    class Emergency(Section):
        _section_name_ = SectionType.EMERGENCY.value
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.EMERGENCY, capacity)
        
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
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.WARD, capacity)
        
        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.OUTSIDE
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert ward-specific processing here
                await asyncio.sleep(1)

    class CCU(Section):
        _section_name_ = SectionType.CCU.value
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.CCU, capacity)

        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.WARD
            if self.dist.need_for_resurgery_after_complex_surgery():
                patient.patient_type == PatientType.NON_ELECTIVE
                next_section_type = SectionType.OPERATING_ROOMS 
                patient.re_surgery_times += 1
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert CCU-specific processing here
                await asyncio.sleep(1)

    class ICU(Section):
        _section_name_ = SectionType.ICU.value
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.ICU, capacity)

        def decide_next_section(self, patient: Patient):
            next_section_type = SectionType.WARD
            if self.dist.need_for_resurgery_after_complex_surgery():
                patient.patient_type == PatientType.NON_ELECTIVE
                next_section_type = SectionType.OPERATING_ROOMS
                patient.re_surgery_times += 1
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert ICU-specific processing here
                await asyncio.sleep(1)

    class Labratory(Section):
        _section_name_ = SectionType.LABRATORY.value
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.LABRATORY, capacity)

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
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.PRE_SURGERY, capacity)

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
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.OPERATING_ROOMS, capacity)

        def decide_next_section(self, patient: Patient):
            if patient.surgery_type == SurgeryType.SIMPLE:
                next_section_type = SectionType.WARD
            elif patient.surgery_type == SurgeryType.MEDIOCRE:
                next_section_type = self.dist.mediocre_surgery_transfer_section()
            else: # COMPLEX                
                next_section_type = self.dist.complex_surgery_transfer_section(patient=patient)
            return next_section_type

        async def section_process(self):
            while self.running:
                # Insert operating room-specific processing here
                await asyncio.sleep(1)
            

    def __init__(self):
        self.emergency = self.Emergency(SECTION_CAPACITIES.get(SectionType.EMERGENCY), )
        self.ward = self.Ward(SECTION_CAPACITIES.get(SectionType.WARD), )
        self.ccu = self.CCU(SECTION_CAPACITIES.get(SectionType.CCU), )
        self.icu = self.ICU(SECTION_CAPACITIES.get(SectionType.ICU), )
        self.operating_rooms = self.OperatingRooms(SECTION_CAPACITIES.get(SectionType.OPERATING_ROOMS), )
        self.labratory = self.Labratory(SECTION_CAPACITIES.get(SectionType.LABRATORY), )
        self.pre_surgery = self.PreSurgery(SECTION_CAPACITIES.get(SectionType.PRE_SURGERY), )

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
        print("reducing")
        await self.icu.reduce_capacity(reduction_percent=REDUCTION_PERCENT)
        await self.ccu.reduce_capacity(reduction_percent=REDUCTION_PERCENT)
        event = Event(EventType.PC_PD, _time_ = (time.time()- st)* 3600/ SIMULATION_SPEED, patient=None)
        Section.all_events.append(event)
        print("[Hospital] Power Outage Event!")
        

class ClientGeneratorForHospital(Section):
    def __init__(self, targeted_hospital: Hospital, dist: Distribiutions, capacity: Capacity):
        self.section_type = SectionType.OUTSIDE
        self.capacity = capacity
        self.entities: List[Patient] = []
        self.queue = asyncio.Queue(maxsize=capacity.queue) if capacity.queue is not None else asyncio.Queue()

        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Initially not paused

        self.workers: List[asyncio.Task] = []
        self.worker_to_patient = {}
        
        Section.__instances__[self.section_type.value] = self

        self.queue_lock = asyncio.Lock()
        self.patients_to_process = []


        self.targeted_hospital = targeted_hospital
        self.dist = dist
        self.running = True
        self.pause = False

        Section.__instances__[self.section_type.value] = self
        
        self.queue_size_time_series = ([], [])
        self.entity_size_time_series = ([], [])
        self.duration_serve = []

    async def run(self, distrib: Distribiutions):
        print("[ClientGenerator] Workers started.")
        self.start_workers(self.dist)
        while self.running:
            await self.pause_event.wait()
            await asyncio.sleep(0.1 / SIMULATION_SPEED) 


    async def run_patient_generator(self):
        print("[ClientGenerator] Started.")
        # count = 0
        while self.running:
            next_patients_type = self.dist.generate_next_patient_type()
            next_patient_interval = self.dist.generate_next_patient_time(patient_type=next_patients_type)
            list_of_patients = self.dist.generate_next_group_of_patients(next_patients_type)
            print(f"[ClientGenerator] New patients in {next_patient_interval} with len {len(list_of_patients)} and type: {next_patients_type.value}")
            
            # Simulate patient arrival interval
            await asyncio.sleep(next_patient_interval * 3600 / SIMULATION_SPEED)
            
            for p in list_of_patients:
                self.entities.append(p)
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
                # response = self.targeted_hospital.emergency.request_handler(message=request)
                response = await self.request_sender(message=request)
                if response.status == ResponseStatus.ACCEPTED:
                    self.all_patients.append(p)
                print(f"[ClientGenerator] Sent entry request for patient: {p.id} - Response: {response.status.value}")
            # count += 1
            # if count == 1:
            #     await self.stop_all()

    async def stop_all(self):
        self.running = False
        # await self.stop()
        print("[ClientGenerator] Stopping...")
        print("[ClientGenerator] Stopped.")

    def decide_next_section(self, patient):
        return super().decide_next_section(patient)
    
    async def section_process(self):
        return await super().section_process()

class Nature:
    class RIP(Section):
        _section_name_ = SectionType.RIP.value
        def __init__(self, capacity: Capacity) -> None:
            super().__init__(SectionType.RIP, capacity)
        def decide_next_section(self, patient):
            return super().decide_next_section(patient)
        def section_process(self):
            return super().section_process()
        pass

    def __init__(self, targeted_hospital: Hospital, dist: Distribiutions,):
        self.running = True
        self.hospital = targeted_hospital
        self.dist = dist
        self.rip = self.RIP(SECTION_CAPACITIES.get(SectionType.RIP))


    async def run(self):
        print("[Nature] Started.")
        power_outage_day = self.dist.uniform_dist(1, 30)
        # power_outage_day = 1
        while self.running:
            global SIMULATION_CLOCK
            clock = SIMULATION_CLOCK
            days, remainder = divmod(int(clock), 86400)  # 86400 seconds in a day
            day_in_this_month = days % 30
            day_outage = days + power_outage_day
            # print(f"[Nature] Next Power outage is on day {day_outage} / now is day {days + day_in_this_month}")
            if day_in_this_month != power_outage_day:
                await asyncio.sleep(3 * 3600 / SIMULATION_SPEED)
                continue
            # self.hospital.electricity_suspension()
            pass

    async def stop(self):
        self.running = False
        print("[Nature] Stopping...")
        print("[Nature] Stopped.")