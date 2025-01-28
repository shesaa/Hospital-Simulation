from hospital_base import *
import json



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


import dash_bootstrap_components as dbc
import plotly.graph_objs as go


# Initialize the simulation
simulation = Simulation()

# Initialize Dash app with Bootstrap theme
app_dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app_dash.title = "Hospital Simulation Dashboard"

def generate_section_graph(section_name):
    return dbc.Card(
        [
            dbc.CardHeader(html.H5(f"{section_name} Section", style={'color': '#3498db'})),
            dbc.CardBody(
                dcc.Graph(
                    id=f'graph-{section_name.lower()}',
                    config={'displayModeBar': False},
                    style={
                        'height': '300px', 
                        'width': '100%', 
                        'border': '2px solid #7f8c8d', 
                        'borderRadius': '10px', 
                        'backgroundColor': '#2c3e50'
                    }
                )
            )
        ],
        style={'width': '100%', 'margin': '10px'}
    )

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
                                    dbc.Button("Pause", id=f'pause-{section_type.value.lower()}', color="danger", className="me-2", n_clicks=0, disabled=False),
                                    dbc.Button("Resume", id=f'resume-{section_type.value.lower()}', color="success", className="me-2", n_clicks=0, disabled=True),
                                ], className="text-center")
                            ],
                            className="mb-3",
                            style={'backgroundColor': '#2c3e50', 'border': '1px solid #7f8c8d'}
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
        
        # Patient Visualization
        dbc.Card(
            dbc.CardBody([
                html.H3("Patient Visualization", className="text-center text-light"),
                dbc.Row([
                    dbc.Col(
                        generate_section_graph(section_type.value),
                        width=6,
                        md=4
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
                    text=f"ID: {patient.id}<br>Type: {patient.patient_type}<br>Surgery: {patient.surgery_type.value}"
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
                    text=f"ID: {patient.id}<br>Type: {patient.patient_type}<br>Surgery: {patient.surgery_type.value}"
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

@app_dash.callback(
    Output('dummy-output', 'children'),  # Dummy output since Dash requires an output
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

@app_dash.callback(
    Output('dummy-output-speed', 'children'),  # Dummy output
    Input('speed-slider', 'value')
)
def update_speed(value):
    global SIMULATION_SPEED
    SIMULATION_SPEED = value
    # Optionally, you can log or handle speed changes here
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
    simulation_task = asyncio.create_task(simulation.run_simulation(duration=120))  # Run for 120 seconds

    # Wait for the simulation to finish
    await simulation_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")