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

from utils import *
from comminucation import *

from time import time
st = time()
# Define Enums and Classes



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
    }
}


# Define the Section and Hospital Classes
class Section(ABC):
    LEDGER_QUEUE = True
    __instances__: Dict[str, 'Section'] = {}

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

    def request_handler(self, message: Request) -> Response:
        print(f"[{self.section_type.value}] Message received from {message.section_req.value} / message : {message.section_from.value} to {message.section_to.value}/ type: {message._type_} / patient id : {message.patient.id}")
        if message._type_ == RequestType.CLIENT_TO_SERVER:
            # Handling client to server requests (e.g., adding to queue)
            try:
                self.queue.put_nowait(message.patient)
                print(f"[{self.section_type.value}] Patient {message.patient.id} added to queue.")
                simulation_state[self.section_type.value]["queue"].append(message.patient.to_dict())
                return Response(request=message, status=ResponseStatus.ACCEPTED)
            except asyncio.QueueFull:
                print(f"[{self.section_type.value}] Queue full. Patient {message.patient.id} rejected.")
                return Response(request=message, status=ResponseStatus.REJECTED)
        else:
            # Handling server to client requests (e.g., moving patient out)
            print(30*"%")
            print("Handling server to client requests (e.g., moving patient out)")
            print(self.section_type, "section_type")
            print(message.patient.section, "message.patient.section")
            print(message.section_req, "section_req")
            print(message.section_from, "section_from")
            print(message.section_to, "section_to")
            print(30*"%")
            if message.patient in self.entities:
                print("here4")
                print(f"patient id {message.patient.id} to {message.section_to}")
                self.move_patient(patient=message.patient, target_section=message.section_to)
                # self.entities.remove(message.patient)
                # print(f"[{self.section_type.value}] Patient {Patient.id} removed from entities.")
                # simulation_state[self.section_type.value]["entities"].remove(message.patient.to_dict())
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
            # self.simulation_state[self.section_type.value]["queue"].append(message.patient.to_dict())
            pass
        elif response.status == ResponseStatus.REJECTED:
            pass  # Handle rejection if needed
        elif response.status == ResponseStatus.SENT:
            # Move patient to target section
            target_section = Section.__instances__.get(message.section_to.value)
            if target_section:
                # message.patient.section = target_section.section_type
                # target_section.entities.append(message.patient)
                # self.simulation_state[self.section_type.value]["entities"].remove(message.patient.to_dict())
                # self.simulation_state[message.section_to.value]["entities"].append(message.patient.to_dict())
                # self.simulation_state[self.section_type.value]["entities"].append(message.patient.to_dict())
                # self.move_patient(patient=message.patient, target_section=target_section)
                pass
        return response

    @abstractmethod
    async def run(self, distrib: Distribiutions):
        '''
        All the operations expected to be committed inside a section
        '''
        pass

    @abstractmethod
    async def stop(self):
        pass

    async def pause(self):
        self.pause_event.clear()
        print(f"[{self.section_type.value}] Paused.")

    async def resume(self):
        self.pause_event.set()
        print(f"[{self.section_type.value}] Resumed.")

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
                patient = await self.queue.get()
                print("----------"* 6 , "Block")
                
                move_request = Request(
                        section_req= self.section_type,
                        section_from=patient.section,
                        section_to=self.section_type,
                        patient=patient,
                        _type_=RequestType.SERVER_TO_CLIENT
                    )
                
                response = await self.request_sender(move_request)
                print("block_time:", time()- st)
                # self.entities.append(patient)
                # print("BEFORE")
                # print(simulation_state)
                # simulation_state[self.section_type.value]["entities"].append(patient.to_dict())
                # simulation_state[self.section_type.value]["queue"].remove(patient.to_dict())
                # print("AFTER")
                # print(simulation_state)
                print(f"patient {patient.id} 's moved from queue into the {self.section_type.value} entities")
                print(f"[{self.section_type.value}] Worker {worker_id} serving patient: {patient.id}")

                # Simulate serving time
                serve_time = distrib.generate_service_time(patient)
                print(f"serve_time for patient {patient.id}: {serve_time}")
                s = time()
                await asyncio.sleep(serve_time)
                print(f"[{self.section_type.value}] Worker {worker_id} finished serving patient: {patient.id}")
                print("duration :", time() - s)

                # Optionally move patient to another section or discharge
                if self.section_type == SectionType.EMERGENCY:
                    section_to = SectionType.WARD
                    move_request = Request(
                        section_req= self.section_type,
                        section_from=SectionType.EMERGENCY,
                        section_to=section_to,
                        patient=patient,
                        _type_=RequestType.CLIENT_TO_SERVER
                    )
                    response = await self.request_sender(move_request)
                    is_moved = await self.wait_for_section(patient=patient, specific_section=section_to)
                    if is_moved:
                        print(f"[{self.section_type.value}] Worker {worker_id} moved patient {patient.id} to Ward.")
                elif self.section_type == SectionType.WARD:
                    # Handle discharge or other logic
                    print(f"[{self.section_type.value}] Patient {patient.id} discharged.")
                    # Remove patient from ward
                    self.entities.remove(patient)
                    simulation_state[self.section_type.value]["entities"].remove(patient.to_dict())
                    patient.section = SectionType.OUTSIDE
                    
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
        print("state BEFORE update")
        print(simulation_state)
        # Update patient's section

        self.entities.remove(patient)
        
        simulation_state[self.section_type.value]["entities"].remove(patient.to_dict())
        target_section_instance = Section.__instances__.get(target_section.value)
        simulation_state[target_section.value]["queue"].remove(patient.to_dict())
        patient.section = target_section
        target_section_instance.entities.append(patient)
        simulation_state[target_section.value]["entities"].append(patient.to_dict())
        print("hahaha")

        


        print(f"[{self.section_type.value}] Patient {patient.id} moved to {target_section.value}")
        print("state AFTER update")
        print(simulation_state)

    async def wait_for_section(self, patient: Patient, specific_section: SectionType, timeout: float = 100):
        '''
        Waits until the patient's section matches the specific section or until timeout.
        '''
        start_time = time.time()
        while patient.section != specific_section:
            if time.time() - start_time > timeout:
                print(f"Timeout: Patient {patient.id} did not move to {specific_section.value} within {timeout} seconds.")
                return False
            await asyncio.sleep(0.1)  # Wait for 100ms before checking again
        return True

class Hospital:
    class Emergency(Section):
        LEDGER_QUEUE = False

        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.EMERGENCY, capacity, simulation_state)

        async def run(self, distrib: Distribiutions):
            print("[Emergency] Section started.")
            self.start_workers(distrib)
            while self.running:
                await self.pause_event.wait()
                await asyncio.sleep(0.1)  # Idle loop; actual processing is done by workers

        async def stop(self):
            self.running = False
            for worker in self.workers:
                worker.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
            print("[Emergency] Section stopped.")

    class Ward(Section):
        def __init__(self, capacity: Capacity, simulation_state: Dict) -> None:
            super().__init__(SectionType.WARD, capacity, simulation_state)

        async def run(self, distrib: Distribiutions):
            print("[Ward] Section started.")
            self.start_workers(distrib)
            while self.running:
                await self.pause_event.wait()
                await asyncio.sleep(0.1)  # Idle loop; actual processing is done by workers

        async def stop(self):
            self.running = False
            for worker in self.workers:
                worker.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
            print("[Ward] Section stopped.")

    def __init__(self, simulation_state: Dict):
        self.emergency = self.Emergency(Capacity(servers=3, queue=10), simulation_state)
        self.ward = self.Ward(Capacity(servers=2, queue=None), simulation_state)
        self.sections = {
            SectionType.EMERGENCY: self.emergency,
            SectionType.WARD: self.ward
        }

    def get_section(self, section_type: SectionType) -> Optional[Section]:
        return self.sections.get(section_type)

class ClientGeneratorForHospital(Section):
    def __init__(self, targeted_hospital: Hospital, dist: Distribiutions, simulation_state: Dict):
        self.section_type = SectionType.OUTSIDE
        self.entities: List[Patient] = []
        self.targeted_hospital = targeted_hospital
        self.dist = dist
        self.running = True
        self.patient_id = 0  # Unique identifier for patients
        self.simulation_state = simulation_state
        Section.__instances__[self.section_type.value] = self

    async def run(self):
        print("[ClientGenerator] Started.")
        while self.running:
            next_patient_type = self.dist.generate_next_patient_type()
            # print("hey,", next_patient_type)
            next_patient_interval = self.dist.generate_next_patient_time(patient_type=next_patient_type)
            self.patient_id += 1
            new_patient = Patient(
                id=self.patient_id,
                patient_type=next_patient_type,
                surgery_type=SurgeryType.SIMPLE,
                section= SectionType.OUTSIDE
            )
            self.entities.append(new_patient)
            simulation_state["OUTSIDE"]["entities"].append(new_patient.to_dict())
            print(f"[ClientGenerator] Generated new patient: {new_patient.id} (Type: {next_patient_type.value}) in {next_patient_interval}s")
            await asyncio.sleep(next_patient_interval)  # Simulate patient arrival interval

            # Create and send request
            request = Request(
                section_req = self.section_type,
                section_from=SectionType.OUTSIDE,
                section_to=SectionType.EMERGENCY,
                patient=new_patient,
                _type_=RequestType.CLIENT_TO_SERVER
            )
            # print("here2")
            response = self.targeted_hospital.emergency.request_handler(message=request)
            # response = await self.request_sender(message=request)
            # if response.status == ResponseStatus.ACCEPTED:
            #     simulation_state["EMERGENCY"]["queue"].append(new_patient.to_dict())
            print(f"[ClientGenerator] Sent entry request for patient: {new_patient.id} - Response: {response.status.value}")

    async def stop(self):
        self.running = False
        print("[ClientGenerator] Stopping...")
        print("[ClientGenerator] Stopped.")

# ---------------------------
