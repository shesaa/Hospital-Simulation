from hospital_base import *
import json
import pandas as pd


@dataclass
class SimulationStateData:
    patients: Dict[str, Dict] = field(default_factory=lambda: {
        section.value: {"entities": [], "queue": []} for section in SectionType
    })
    
    # Initialize a sections_status dictionary setting each section to True.
    sections_status: Dict[str, bool] = field(default_factory=lambda: {
        section.value: True for section in SectionType
    })

class Simulation:
    def __init__(self):
        self.hospital = Hospital()
        self.dist = Distribiutions(cnfg=None)
        self.client_generator = ClientGeneratorForHospital(
            targeted_hospital=self.hospital,
            dist=self.dist,
            capacity= SECTION_CAPACITIES.get(SectionType.OUTSIDE)
        )
        # self.state = simulation_state
        self.tasks: List[asyncio.Task] = []

    async def start(self):
        # Start hospital sections
        self.tasks.append(asyncio.create_task(self.hospital.emergency.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.ward.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.labratory.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.pre_surgery.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.operating_rooms.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.icu.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.ccu.run(self.dist)))
        
        # Start client generator
        self.tasks.append(asyncio.create_task(self.client_generator.run()))
        # Start state updater
        self.tasks.append(asyncio.create_task(self.update_state()))
        print("[Simulation] Started.")

    async def update_state(self):
        global SIMULATION_CLOCK
        update_interval = 1  # seconds (real time)
        while True:
            await asyncio.sleep(update_interval)  # Wait for the update interval
            # Increment simulation clock based on speed and interval
            SIMULATION_CLOCK += update_interval * SIMULATION_SPEED

    async def stop(self):
        print("[Simulation] Stopping...")
        await self.client_generator.stop()
        await self.hospital.emergency.stop()
        await self.hospital.pre_surgery.stop()
        await self.hospital.labratory.stop()
        await self.hospital.ccu.stop()
        await self.hospital.icu.stop()
        await self.hospital.operating_rooms.stop()
        await self.hospital.ward.stop()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        # with open('state.json', "w") as _:
        #     json.dump(simulation_state, _, indent=4)
        # print(simulation_state)
        print("[Simulation] Stopped.")

    async def run_simulation(self, duration: int):
        await self.start()
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def pause_section(self, section_type: SectionType):
        section = self.hospital.get_section(section_type)
        if section:
            await section.pause()
            # self.state["sections_status"][section_type.value] = False

    async def resume_section(self, section_type: SectionType):
        section = self.hospital.get_section(section_type)
        if section:
            await section.resume()
            # self.state["sections_status"][section_type.value] = True








# dashboard.py

# dashboard.py


# dashboard.py

import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the simulation
simulation = Simulation()

# Initialize Dash app with Bootstrap theme
app_dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app_dash.title = "Hospital Simulation Dashboard"

def generate_section_card(section_type):
    """
    Generates a Bootstrap card for a given section with nested tabs:
    - Outer Tabs: "Patients" and "Worker Assignments"
    - Inner Tabs within "Worker Assignments": One tab per worker
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Patients", tab_id="patients"),
                        dbc.Tab(label="Worker Assignments", tab_id="workers"),
                    ],
                    id=f'tabs-outer-{section_type.value.lower()}',
                    active_tab="patients",
                )
            ),
            dbc.CardBody(
                html.Div(id=f'content-outer-{section_type.value.lower()}')
            )
        ],
        style={
            'width': '100%', 
            'margin': '10px',
            'border': '2px solid #7f8c8d',
            'borderRadius': '10px',
            'backgroundColor': '#2c3e50'
        }
    )

def generate_worker_tabs(section_type):
    """
    Generates inner tabs for each worker within a section.
    Each inner tab displays the patient the worker is serving.
    """
    section = simulation.hospital.get_section(section_type)
    if not section or not section.worker_to_patient:
        return dbc.Alert("No workers assigned to this section.", color="warning")
    
    workers = section.worker_to_patient.keys()
    tabs = []
    for worker_id in workers:
        tab_id = f'worker-{section_type.value.lower()}-{worker_id}'
        tabs.append(dbc.Tab(label=f"Worker {worker_id}", tab_id=tab_id))
    
    return dbc.Tabs(
        tabs,
        id=f'tabs-inner-{section_type.value.lower()}',
        active_tab=f'worker-{section_type.value.lower()}-{list(workers)[0]}' if workers else None,
    )

def generate_worker_content(section_type, worker_id):
    """
    Generates the content for a worker's inner tab.
    Displays the patient the worker is currently serving.
    """
    section = simulation.hospital.get_section(section_type)
    if not section:
        return dbc.Alert("Section not found.", color="danger")
    
    patient = section.worker_to_patient.get(worker_id)

    if patient:
        patient_info = {
            "Patient ID": patient.id,
            "Type": patient.patient_type.value,
            "Surgery": patient.surgery_type.value
        }
        df = pd.DataFrame([patient_info])
        return dbc.Table.from_dataframe(
            df,
            striped=True,
            bordered=True,
            hover=True,
            dark=True,
            responsive=True
        )
    else:
        return dbc.Alert("No patient assigned.", color="info")


# Define Dash layout using Dash Bootstrap Components
app_dash.layout = dbc.Container(
    fluid=True,
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.H1("Hospital Simulation Dashboard", className="text-center text-primary mb-4"),
                width=12
            )
        ),
        
        # Control Panels
        dbc.Card(
            dbc.CardBody([
                html.H3("Control Panels", className="text-center text-light"),
                dbc.Row([
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H5(f"{section_type.value} Section", className="text-center text-primary")),
                                dbc.CardBody([
                                    dbc.Button("Pause", id=f'pause-{section_type.value.lower()}', color="danger", className="me-2 mb-2", n_clicks=0, disabled=False),
                                    dbc.Button("Resume", id=f'resume-{section_type.value.lower()}', color="success", className="me-2 mb-2", n_clicks=0, disabled=True),
                                ], className="text-center")
                            ],
                            className="mb-3",
                            style={'backgroundColor': '#34495e', 'border': '1px solid #7f8c8d'}
                        ),
                        width=3
                    ) for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
                ])
            ]),
            className="mb-4",
            style={'backgroundColor': '#34495e', 'border': 'none'}
        ),
        
        # Live Status Display
        dbc.Card(
            dbc.CardBody([
                html.H3("Live Status", className="text-center text-light"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("Section", className="text-center"),
                                html.Th("Entities", className="text-center"),
                                html.Th("Queue Length", className="text-center")
                            ], style={'backgroundColor': '#2980b9', 'color': 'white'})
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(section_type.value, className="text-center"),
                                html.Td(id=f'entities-{section_type.value.lower()}', className="text-center"),
                                html.Td(id=f'queue-{section_type.value.lower()}', className="text-center")
                            ]) for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
                        ])
                    ],
                    bordered=True,
                    dark=True,
                    hover=True,
                    responsive=True,
                    striped=True
                )
            ]),
            className="mb-4",
            style={'backgroundColor': '#34495e', 'border': 'none'}
        ),
        
        # Patient Visualization with Inner Tabs
        dbc.Card(
            dbc.CardBody([
                html.H3("Patient Visualization", className="text-center text-light"),
                dbc.Row([
                    dbc.Col(
                        generate_section_card(section_type),
                        width=12,
                        md=6
                    ) for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
                ])
            ]),
            className="mb-4",
            style={'backgroundColor': '#34495e', 'border': 'none'}
        ),
        
        # Simulation Clock
        dbc.Row(
            dbc.Col(
                dbc.Alert(id='clock-display', color="info", className="text-center"),
                width=12
            )
        ),
        
        # Interval Component for Callbacks
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # 1 second in milliseconds
            n_intervals=0
        )
    ]
)

# Callback to update live status numbers
@app_dash.callback(
    [
        Output(f'entities-{section_type.value.lower()}', 'children') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ] +
    [
        Output(f'queue-{section_type.value.lower()}', 'children') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ],
    Input('interval-component', 'n_intervals')
)
def update_section_statuses(n_intervals):
    statuses_entities = []
    statuses_queue = []
    for section_type in SectionType:
        if section_type in [SectionType.OUTSIDE, SectionType.RIP]:
            continue
        section = simulation.hospital.get_section(section_type)
        if section:
            entities_count = len(section.entities)
            queue_count = section.queue.qsize()
            statuses_entities.append(str(entities_count))
            statuses_queue.append(str(queue_count))
        else:
            statuses_entities.append("0")
            statuses_queue.append("0")
    return statuses_entities + statuses_queue

# Callback to update patient visualization graphs
@app_dash.callback(
    [Output(f'graph-{section_type.value.lower()}', 'figure') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]],
    Input('interval-component', 'n_intervals')
)
def update_graphs(n_intervals):
    figures = []
    MAX_DISPLAY = 20  # Maximum number of patients to display per section
    for section_type in SectionType:
        if section_type in [SectionType.OUTSIDE, SectionType.RIP]:
            continue
        section = simulation.hospital.get_section(section_type)
        if section:
            entities = section.entities
            queue = list(section.queue._queue)
            
            # Determine if patients exceed the display limit
            extra_queue = len(queue) - MAX_DISPLAY
            extra_entities = len(entities) - MAX_DISPLAY
            
            displayed_queue = queue[:MAX_DISPLAY]
            displayed_entities = entities[:MAX_DISPLAY]
            
            # Prepare data for queued patients (blue)
            queue_traces = []
            for idx, patient in enumerate(displayed_queue):
                queue_traces.append(go.Scatter(
                    x=[idx],
                    y=[0],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='blue'),
                    name='Queued Patient',
                    hoverinfo='text',
                    text=f"ID: {patient.id}<br>Type: {patient.patient_type.value}<br>Surgery: {patient.surgery_type.value}"
                ))
            
            # Prepare data for serving patients (red)
            entity_traces = []
            for idx, patient in enumerate(displayed_entities):
                entity_traces.append(go.Scatter(
                    x=[idx],
                    y=[1],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='red'),
                    name='Serving Patient',
                    hoverinfo='text',
                    text=f"ID: {patient.id}<br>Type: {patient.patient_type.value}<br>Surgery: {patient.surgery_type.value}"
                ))
            
            # Combine traces
            data = queue_traces + entity_traces
            
            # Add annotations for extra patients if any
            annotations = []
            if extra_queue > 0:
                annotations.append(dict(
                    x=MAX_DISPLAY - 1,
                    y=0,
                    text=f"+{extra_queue} more queued",
                    showarrow=False,
                    font=dict(color='white', size=12)
                ))
            if extra_entities > 0:
                annotations.append(dict(
                    x=MAX_DISPLAY - 1,
                    y=1,
                    text=f"+{extra_entities} more serving",
                    showarrow=False,
                    font=dict(color='white', size=12)
                ))
            
            # Define layout without axes and with fixed y-axis
            layout = go.Layout(
                title=f"{section_type.value} Section",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 1.5]),
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='#2c3e50',
                plot_bgcolor='#2c3e50',
                height=200,
                annotations=annotations
            )
            
            fig = go.Figure(data=data, layout=layout)
            figures.append(fig)
        else:
            # If section not found, append an empty figure
            figures.append(go.Figure())
    return figures

# Callback to update outer tab content (Patients or Worker Assignments)
@app_dash.callback(
    [
        Output(f'content-outer-{section_type.value.lower()}', 'children') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ],
    [
        Input(f'tabs-outer-{section_type.value.lower()}', 'active_tab') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ] +
    [
        Input('interval-component', 'n_intervals')
    ]
)
def update_outer_tabs(*args):
    """
    Updates the content within each section's outer tabs.
    - If "Patients" tab is active, display the patient graph.
    - If "Worker Assignments" tab is active, display inner tabs for workers.
    """
    n_intervals = args[-1]
    active_tabs_outer = args[:-1]
    contents = []
    
    for idx, section_type in enumerate(SectionType):
        if section_type in [SectionType.OUTSIDE, SectionType.RIP]:
            continue
        active_tab = active_tabs_outer[idx]
        if active_tab == "patients":
            # Display the patient graph
            contents.append(
                dcc.Graph(
                    id=f'graph-inner-{section_type.value.lower()}',
                    figure=update_graphs(n_intervals)[idx],
                    config={'displayModeBar': False},
                    style={
                        'height': '200px', 
                        'width': '100%', 
                        'border': '2px solid #7f8c8d', 
                        'borderRadius': '10px', 
                        'backgroundColor': '#2c3e50'
                    }
                )
            )
        elif active_tab == "workers":
            # Generate inner tabs for workers
            inner_tabs = generate_worker_tabs(section_type)
            if isinstance(inner_tabs, dbc.Alert):
                contents.append(inner_tabs)
            else:
                contents.append(
                    html.Div([
                        inner_tabs,
                        html.Div(id=f'content-inner-{section_type.value.lower()}')
                    ])
                )
        else:
            contents.append(html.P("Select a tab to view content."))
    
    return contents

# Callback to update worker assignments content based on inner active tabs
@app_dash.callback(
    [
        Output(f'content-inner-{section_type.value.lower()}-{worker_id}', 'children') 
        for section_type in SectionType 
        if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
        for worker_id in simulation.hospital.get_section(section_type).worker_to_patient.keys()
    ],
    [
        Input(f'tabs-inner-{section_type.value.lower()}', 'active_tab') 
        for section_type in SectionType 
        if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
        for _ in simulation.hospital.get_section(section_type).worker_to_patient.keys()
    ] +
    [
        Input('interval-component', 'n_intervals')
    ]
)
def update_worker_contents(*args):
    """
    Updates the content within each worker's inner tab.
    Displays the patient the worker is currently serving.
    """
    n_intervals = args[-1]
    active_tabs_inner = args[:-1]
    contents = []
    
    # Iterate over sections and workers to update their content
    worker_idx = 0
    for section_type in SectionType:
        if section_type in [SectionType.OUTSIDE, SectionType.RIP]:
            continue
        section = simulation.hospital.get_section(section_type)
        if not section:
            continue
        for worker_id in section.worker_to_patient.keys():
            active_tab = active_tabs_inner[worker_idx]
            expected_tab_id = f'worker-{section_type.value.lower()}-{worker_id}'
            if active_tab == expected_tab_id:
                patient = section.worker_to_patient.get(worker_id)
                if patient:
                    patient_info = {
                        "Patient ID": patient.id,
                        "Type": patient.patient_type.value,
                        "Surgery": patient.surgery_type.value
                    }
                    df = pd.DataFrame([patient_info])
                    contents.append(
                        dbc.Table.from_dataframe(
                            df,
                            striped=True,
                            bordered=True,
                            hover=True,
                            dark=True,
                            responsive=True
                        )
                    )

                else:
                    contents.append(
                        dbc.Alert("No patient assigned.", color="info")
                    )
            else:
                contents.append(html.P("Select a worker tab to view details."))
            worker_idx += 1
    
    return contents

# Callback to update simulation clock
@app_dash.callback(
    Output('clock-display', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_clock(n):
    global SIMULATION_CLOCK
    days, remainder = divmod(int(SIMULATION_CLOCK), 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"Simulation Time: {days}d {hours:02}h {minutes:02}m {seconds:02}s"
    return dbc.Alert(time_str, color="info", className="text-center")

# Callback to handle pause and resume buttons
@app_dash.callback(
    [
        Output(f'pause-{section_type.value.lower()}', 'disabled') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ] +
    [
        Output(f'resume-{section_type.value.lower()}', 'disabled') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ],
    [
        Input(f'pause-{section_type.value.lower()}', 'n_clicks') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ] +
    [
        Input(f'resume-{section_type.value.lower()}', 'n_clicks') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ]
)
def update_button_states(*args):
    num_sections = (len(SectionType) - 2)  # Excluding OUTSIDE and RIP
    pause_clicks = args[:num_sections]
    resume_clicks = args[num_sections:]
    
    # Initialize lists for button states
    pause_disabled = []
    resume_disabled = []
    
    for idx, section_type in enumerate(SectionType):
        if section_type in [SectionType.OUTSIDE, SectionType.RIP]:
            continue
        section = simulation.hospital.get_section(section_type)
        if section:
            # If section is paused, disable pause button and enable resume
            if section.paused:
                pause_disabled.append(True)
                resume_disabled.append(False)
            else:
                pause_disabled.append(False)
                resume_disabled.append(True)
        else:
            # Default states
            pause_disabled.append(True)
            resume_disabled.append(True)
    
    return pause_disabled + resume_disabled

# Callback to handle button clicks for pausing and resuming sections
@app_dash.callback(
    Output('dummy-output', 'children'),  # Dummy output since Dash requires at least one output
    [
        Input(f'pause-{section_type.value.lower()}', 'n_clicks') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ] +
    [
        Input(f'resume-{section_type.value.lower()}', 'n_clicks') for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]
    ]
)
def control_sections(*args):
    num_sections = (len(SectionType) - 2)  # Excluding OUTSIDE and RIP
    pause_clicks = args[:num_sections]
    resume_clicks = args[num_sections:]
    
    for idx, section_type in enumerate(SectionType):
        if section_type in [SectionType.OUTSIDE, SectionType.RIP]:
            continue
        section = simulation.hospital.get_section(section_type)
        if section:
            # If pause button clicked
            if pause_clicks[idx] > 0:
                asyncio.create_task(simulation.pause_section(section_type))
            
            # If resume button clicked
            if resume_clicks[idx] > 0:
                asyncio.create_task(simulation.resume_section(section_type))
    
    return ""

# Callback to handle simulation speed slider (if implemented)
@app_dash.callback(
    Output('dummy-output-speed', 'children'),  # Dummy output
    Input('speed-slider', 'value')
)
def update_speed(value):
    global SIMULATION_SPEED
    SIMULATION_SPEED = value
    logger.info(f"Simulation speed updated to: {SIMULATION_SPEED}")
    return ""

# Function to run Dash in a separate thread
def run_dash():
    app_dash.run_server(debug=False, use_reloader=False, port=8060)

# Main Async Function
async def main():
    # Start Dash in a separate thread
    dash_thread = threading.Thread(target=run_dash, daemon=True)
    dash_thread.start()

    # Start the simulation
    simulation_task = asyncio.create_task(simulation.run_simulation(duration=SIMULATION_DURATION))  # Run for 1 hour

    # Wait for the simulation to finish
    await simulation_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")