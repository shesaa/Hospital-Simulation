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
        print(f"[{self.section_type.value}] Message received from {message.section_from} to {message.section_to} / type: {message._type_}")
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
            if message.patient in self.entities:
                self.entities.remove(message.patient)
                print(f"[{self.section_type.value}] Patient removed from entities.")
                return Response(request=message, status=ResponseStatus.SENT)
            print(f"[{self.section_type.value}] Patient not found in entities.")
            return Response(request=message, status=ResponseStatus.REJECTED)

    async def request_sender(self, message: Request) -> Response:
        section = Section.__instances__.get(message.section_to.value)
        if not section:
            print(f"Section {message.section_to.value} does not exist.")
            return Response(request=message, status=ResponseStatus.REJECTED)
        response = section.request_handler(message)
        # Update simulation_state
        if response.status == ResponseStatus.ACCEPTED:
            self.simulation_state[self.section_type.value]["queue"].append(message.patient.to_dict())
        elif response.status == ResponseStatus.REJECTED:
            pass  # Handle rejection if needed
        elif response.status == ResponseStatus.SENT:
            # Move patient to target section
            target_section = Section.__instances__.get(message.section_to.value)
            if target_section:
                target_section.entities.append(message.patient)
                self.simulation_state[self.section_type.value]["entities"].remove(message.patient.to_dict())
                self.simulation_state[message.section_to.value]["entities"].append(message.patient.to_dict())
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
        
        '''
        print(f"[{self.section_type.value}] Worker {worker_id} started.")
        
        while self.running:
            await self.pause_event.wait()  # Wait if paused
            try:
                patient = await self.queue.get()
                print("----------"* 6 , "Block")
                
                print("block_time:", time()- st)
                self.entities.append(patient)
                simulation_state[self.section_type.value]["entities"].append(patient.to_dict())
                simulation_state[self.section_type.value]["queue"].remove(patient.to_dict())
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
                    move_request = Request(
                        section_from=SectionType.EMERGENCY,
                        section_to=SectionType.WARD,
                        patient=patient,
                        _type_=RequestType.CLIENT_TO_SERVER
                    )
                    response = await self.request_sender(move_request)
                    if response.status == ResponseStatus.SENT:
                        print(f"[{self.section_type.value}] Worker {worker_id} moved patient {patient.id} to Ward.")
                elif self.section_type == SectionType.WARD:
                    # Handle discharge or other logic
                    print(f"[{self.section_type.value}] Patient {patient.id} discharged.")
                    # Remove patient from ward
                    self.entities.remove(patient)
                    simulation_state[self.section_type.value]["entities"].remove(patient.to_dict())
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
        self.emergency = self.Emergency(Capacity(servers=3, queue=10), simulation_state)  # Example: 3 servers in Emergency
        self.ward = self.Ward(Capacity(servers=2, queue=None), simulation_state)          # Example: 2 servers in Ward
        self.sections = {
            SectionType.EMERGENCY: self.emergency,
            SectionType.WARD: self.ward
        }

    def get_section(self, section_type: SectionType) -> Optional[Section]:
        return self.sections.get(section_type)

class ClientGeneratorForHospital(Section):
    def __init__(self, targeted_hospital: Hospital, dist: Distribiutions, simulation_state: Dict):
        self.targeted_hospital = targeted_hospital
        self.dist = dist
        self.running = True
        self.patient_id = 0  # Unique identifier for patients
        self.simulation_state = simulation_state

    async def run(self):
        print("[ClientGenerator] Started.")
        while self.running:
            next_patient_type = self.dist.generate_next_patient_type()
            print("hey,", next_patient_type)
            next_patient_interval = self.dist.generate_next_patient_time(patient_type=next_patient_type)
            self.patient_id += 1
            new_patient = Patient(
                id=self.patient_id,
                patient_type=next_patient_type,
                surgery_type=SurgeryType.SIMPLE
            )
            print(f"[ClientGenerator] Generated new patient: {new_patient} (Type: {next_patient_type}) in {next_patient_interval}s")
            await asyncio.sleep(next_patient_interval)  # Simulate patient arrival interval

            # Create and send request
            request = Request(
                section_from=SectionType.OUTSIDE,
                section_to=SectionType.EMERGENCY,
                patient=new_patient,
                _type_=RequestType.CLIENT_TO_SERVER
            )
            response = self.targeted_hospital.emergency.request_handler(request)
            if response.status == ResponseStatus.ACCEPTED:
                simulation_state["EMERGENCY"]["queue"].append(new_patient.to_dict())
            print(f"[ClientGenerator] Sent entry request for patient: {new_patient.id} - Response: {response.status.value}")

    async def stop(self):
        self.running = False
        print("[ClientGenerator] Stopping...")
        print("[ClientGenerator] Stopped.")

# ---------------------------
