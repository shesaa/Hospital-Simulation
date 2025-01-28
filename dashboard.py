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
    def __init__(self, simulation_state: Dict):
        self.hospital = Hospital(simulation_state)
        self.dist = Distribiutions(cnfg=None)
        self.client_generator = ClientGeneratorForHospital(
            targeted_hospital=self.hospital,
            dist=self.dist,
            simulation_state=simulation_state
        )
        self.state = simulation_state
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
        print(simulation_state)
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
            self.state["sections_status"][section_type.value] = False

    async def resume_section(self, section_type: SectionType):
        section = self.hospital.get_section(section_type)
        if section:
            await section.resume()
            self.state["sections_status"][section_type.value] = True

# Dash Application Setup

app_dash = dash.Dash(__name__)

app_dash.layout = html.Div(
    style={
        'fontFamily': 'Arial', 
        'backgroundColor': '#34495e', 
        'color': 'white', 
        'padding': '20px'
    }, 
    children=[
        html.H1("Hospital Simulation Dashboard"),
        
        # Tabs component
        dcc.Tabs(id='tabs', value='simulation-tab', children=[
            dcc.Tab(label='Simulation View', value='simulation-tab', children=[
                html.Div(id='clock-display', style={'fontSize': '24px', 'marginBottom': '20px'}),
                dcc.Graph(id='hospital-graph', config={'staticPlot': False}),
                html.Div([
                    html.Label("Simulation Speed:", style={'marginRight': '10px'}),
                    dcc.Slider(
                        id='speed-slider',
                        min=0.1,
                        max=3600,
                        step=0.1,
                        value=1.0,
                        marks={i: f"{i}x" for i in [0.5, 1, 2, 3, 4, 5, 3600]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '50%', 'margin': '20px 0'}),
                html.Div([
                    html.H2("Control Panels"),
                    html.Div([
                        html.H3("Emergency Section"),
                        html.Button("Pause", id='pause-emergency', n_clicks=0),
                        html.Button("Resume", id='resume-emergency', n_clicks=0),
                    ]),
                    html.Div([
                        html.H3("Ward Section"),
                        html.Button("Pause", id='pause-ward', n_clicks=0),
                        html.Button("Resume", id='resume-ward', n_clicks=0),
                    ]),
                ]),
                # New container for section statuses
                html.Div(id='section-statuses', style={'marginTop': '20px'}),
                
                dcc.Interval(
                    id='interval-component',
                    interval=1000,  # 1 second
                    n_intervals=0
                )
            ]),
            dcc.Tab(label='Live Charts', value='charts-tab', children=[
                dcc.Graph(id='average-wait-chart'),
                dcc.Interval(
                    id='chart-interval',
                    interval=1000,  # 1 second
                    n_intervals=0
                )
            ])
        ])
    ]
)

@app_dash.callback(
    Output('section-statuses', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_section_statuses(n_intervals):
    statuses = []
    # Iterate over each section in simulation_state
    for section, data in simulation_state.items():
        # Calculate the count of entities and queue items
        entities_count = len(data.get('entities', []))
        queue_count = len(data.get('queue', []))
        # Create a Div for each section's status
        statuses.append(
            html.Div(
                f"Section {section}: {entities_count} in service, {queue_count} in queue",
                style={'padding': '5px', 'borderBottom': '1px solid #7f8c8d'}
            )
        )
    return statuses

@app_dash.callback(
    Output('clock-display', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_clock(n):
    # Format simulation time as HH:MM:SS for display
    hours, remainder = divmod(int(SIMULATION_CLOCK), 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"Simulation Time: {hours:02}:{minutes:02}:{seconds:02}"
    return time_str

@app_dash.callback(
    Output('speed-slider', 'value'),
    Input('speed-slider', 'value')
)
def update_speed_slider(value):
    global SIMULATION_SPEED
    SIMULATION_SPEED = value  # Update the global simulation speed
    print("new speed", SIMULATION_SPEED, value)
    return value  # Reflect slider position



@app_dash.callback(
    Output('hospital-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph_live(n):
    # Define section positions and sizes for all sections
    sections = {
        "OUTSIDE": {"x": 10, "y": 10, "width": 150, "height": 100},
        "PRE_SURGERY": {"x": 170, "y": 10, "width": 150, "height": 100},
        "EMERGENCY": {"x": 330, "y": 10, "width": 300, "height": 200},
        "LABRATORY": {"x": 10, "y": 120, "width": 150, "height": 100},
        "OPERATING_ROOMS": {"x": 170, "y": 120, "width": 300, "height": 200},
        "WARD": {"x": 480, "y": 120, "width": 300, "height": 200},
        "ICU": {"x": 10, "y": 330, "width": 150, "height": 100},
        "CCU": {"x": 170, "y": 330, "width": 150, "height": 100},
        "RIP": {"x": 330, "y": 330, "width": 150, "height": 100}
    }

    data = []

    # Draw sections as rectangles with labels
    for name, sec in sections.items():
        rect = go.Scatter(
            x=[sec["x"], sec["x"] + sec["width"], sec["x"] + sec["width"], sec["x"], sec["x"]],
            y=[sec["y"], sec["y"], sec["y"] + sec["height"], sec["y"] + sec["height"], sec["y"]],
            mode='lines',
            name=name
        )
        data.append(rect)
        data.append(go.Scatter(
            x=[sec["x"] + 10],
            y=[sec["y"] + 20],
            mode='text',
            text=[name],
            showlegend=False
        ))

    # Draw patients for each section
    for section, patients_list in simulation_state.items():
        if section in sections:
            sec = sections[section]
            x_base = sec["x"] + 10
            y_base = sec["y"] + 30
            x_spacing = 20
            y_spacing = 20
            cols = int((sec["width"] - 20) / x_spacing)

            # Queue Patients as blue circles
            for idx, patient in enumerate(patients_list.get("queue", [])):
                col = idx % cols
                row = idx // cols
                x = x_base + col * x_spacing
                y = y_base + row * y_spacing
                data.append(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='blue'),
                    name=f"Queue Patient {patient.id}",
                    hoverinfo='text',
                    text=f"ID: {patient.id}<br>Type: {patient.patient_type}<br>Surgery: {patient.surgery_type}"
                ))

            # Entities (Serving Patients) as red circles
            for idx, patient in enumerate(patients_list.get("entities", [])):
                col = idx % cols
                row = idx // cols
                x = x_base + col * x_spacing
                y = y_base + row * y_spacing + 100  # Offset for entities
                data.append(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='red'),
                    name=f"Serving Patient {patient.id}",
                    hoverinfo='text',
                    text=f"ID: {patient.id}<br>Type: {patient.patient_type}<br>Surgery: {patient.surgery_type}"
                ))

    layout = go.Layout(
        xaxis=dict(range=[0, 800], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 600], showgrid=False, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

# Control Callbacks

@app_dash.callback(
    Output('pause-emergency', 'n_clicks'),
    Input('pause-emergency', 'n_clicks'),
    prevent_initial_call=True
)
def pause_emergency(n_clicks):
    if n_clicks > 0:
        asyncio.create_task(simulation.pause_section(SectionType.EMERGENCY))
    return 0  # Reset the click count

@app_dash.callback(
    Output('resume-emergency', 'n_clicks'),
    Input('resume-emergency', 'n_clicks'),
    prevent_initial_call=True
)
def resume_emergency(n_clicks):
    if n_clicks > 0:
        asyncio.create_task(simulation.resume_section(SectionType.EMERGENCY))
    return 0

@app_dash.callback(
    Output('pause-ward', 'n_clicks'),
    Input('pause-ward', 'n_clicks'),
    prevent_initial_call=True
)
def pause_ward(n_clicks):
    if n_clicks > 0:
        asyncio.create_task(simulation.pause_section(SectionType.WARD))
    return 0

@app_dash.callback(
    Output('resume-ward', 'n_clicks'),
    Input('resume-ward', 'n_clicks'),
    prevent_initial_call=True
)
def resume_ward(n_clicks):
    if n_clicks > 0:
        asyncio.create_task(simulation.resume_section(SectionType.WARD))
    return 0


# Initialize Simulation
simulation = Simulation(simulation_state)

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


