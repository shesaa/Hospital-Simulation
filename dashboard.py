from hospital_base import *

@dataclass
class SimulationStateData:
    patients: Dict[str, Dict] = field(default_factory=lambda: {
        "EMERGENCY": {"entities": [], "queue": []},
        "WARD": {"entities": [], "queue": []}
    })
    sections_status: Dict[str, bool] = field(default_factory=lambda: {
        "OUTSIDE": True,
        "EMERGENCY": True,
        "WARD": True,
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
        # Start client generator
        self.tasks.append(asyncio.create_task(self.client_generator.run()))
        # Start state updater
        self.tasks.append(asyncio.create_task(self.update_state()))
        print("[Simulation] Started.")

    async def update_state(self):
        while True:
            # Here, simulation_state is already being updated by sections and client generator
            # This function can be used to trigger additional actions or logging if needed
            await asyncio.sleep(1)  # Update interval

    async def stop(self):
        print("[Simulation] Stopping...")
        await self.client_generator.stop()
        await self.hospital.emergency.stop()
        await self.hospital.ward.stop()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
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

app_dash.layout = html.Div([
    html.H1("Hospital Simulation Dashboard"),
    dcc.Graph(id='hospital-graph', config={'staticPlot': False}),
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
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    )
])

@app_dash.callback(
    Output('hospital-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
# def update_graph_live(n):
#     # Define section positions and sizes
#     sections = {
#         "EMERGENCY": {"x": 50, "y": 50, "width": 300, "height": 200},
#         "WARD": {"x": 450, "y": 50, "width": 300, "height": 200},
#         # Add more sections as needed
#     }

#     data = []

#     # Draw sections as rectangles
#     for name, sec in sections.items():
#         rect = go.Scatter(
#             x=[sec["x"], sec["x"] + sec["width"], sec["x"] + sec["width"], sec["x"], sec["x"]],
#             y=[sec["y"], sec["y"], sec["y"] + sec["height"], sec["y"] + sec["height"], sec["y"]],
#             mode='lines',
#             name=name
#         )
#         data.append(rect)
#         # Add section label
#         data.append(go.Scatter(
#             x=[sec["x"] + 10],
#             y=[sec["y"] + 20],
#             mode='text',
#             text=[name],
#             showlegend=False
#         ))

#     # Draw patients as red circles
#     for section, patients_list in simulation_state.items():
#         if section in sections:
#             sec = sections[section]
#             x_base = sec["x"] + 10
#             y_base = sec["y"] + 30
#             x_spacing = 20
#             y_spacing = 20
#             cols = int((sec["width"] - 20) / x_spacing)
#             for idx, patient in enumerate(patients_list["entities"]):
#                 col = idx % cols
#                 row = idx // cols
#                 x = x_base + col * x_spacing
#                 y = y_base + row * y_spacing
#                 data.append(go.Scatter(
#                     x=[x],
#                     y=[y],
#                     mode='markers',
#                     marker=dict(symbol='circle', size=10, color='red'),
#                     name=f"Patient {patient['id']}",
#                     hoverinfo='text',
#                     text=f"ID: {patient['id']}<br>Type: {patient['patient_type']}<br>Surgery: {patient['surgery_type']}"
#                 ))

#     layout = go.Layout(
#         xaxis=dict(range=[0, 800], showgrid=False, zeroline=False),
#         yaxis=dict(range=[0, 600], showgrid=False, zeroline=False),
#         margin=dict(l=20, r=20, t=20, b=20),
#         showlegend=False
#     )

#     fig = go.Figure(data=data, layout=layout)
#     return fig


# Modify the update_graph_live function in Dash

def update_graph_live(n):
    # Define section positions and sizes
    sections = {
        "EMERGENCY": {"x": 50, "y": 50, "width": 300, "height": 200},
        "WARD": {"x": 450, "y": 50, "width": 300, "height": 200},
        # Add more sections as needed
    }

    data = []

    # Draw sections as rectangles
    for name, sec in sections.items():
        rect = go.Scatter(
            x=[sec["x"], sec["x"] + sec["width"], sec["x"] + sec["width"], sec["x"], sec["x"]],
            y=[sec["y"], sec["y"], sec["y"] + sec["height"], sec["y"] + sec["height"], sec["y"]],
            mode='lines',
            name=name
        )
        data.append(rect)
        # Add section label
        data.append(go.Scatter(
            x=[sec["x"] + 10],
            y=[sec["y"] + 20],
            mode='text',
            text=[name],
            showlegend=False
        ))

    # Draw patients waiting in queue as blue circles
    for section, patients_list in simulation_state.items():
        if section in sections:
            sec = sections[section]
            print(40* "*")
            print(section)
            print(simulation_state[section])
            print(40* "*")

            x_base = sec["x"] + 10
            y_base = sec["y"] + 30
            x_spacing = 20
            y_spacing = 20
            cols = int((sec["width"] - 20) / x_spacing)
            # Queue Patients
            for idx, patient in enumerate(patients_list["queue"]):
                # print(40* "*")
                # print(patient)
                # print(type(patient))
                # print(40*"*")
                col = idx % cols
                row = idx // cols
                x = x_base + col * x_spacing
                y = y_base + row * y_spacing
                data.append(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='blue'),
                    name=f"Queue Patient {patient["id"]}",
                    hoverinfo='text',
                    text=f"ID: {patient['id']}<br>Type: {patient['patient_type']}<br>Surgery: {patient['surgery_type']}"
                ))
            # Entities (Serving Patients)
            for idx, patient in enumerate(patients_list["entities"]):
                # print(40* "* entity")
                # print(patient)
                # print(type(patient))
                # print(40*"*entity")
                col = idx % cols
                row = idx // cols
                x = x_base + col * x_spacing
                y = y_base + row * y_spacing + 100  # Offset to separate from queue
                data.append(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='red'),
                    name=f"Serving Patient {patient["id"]}",
                    hoverinfo='text',
                    text=f"ID: {patient['id']}<br>Type: {patient['patient_type']}<br>Surgery: {patient['surgery_type']}"
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

# Initialize Simulation State
# simulation_state = {
#     "EMERGENCY": {
#         "entities": [],
#         "queue": []
#     },
#     "WARD": {
#         "entities": [],
#         "queue": []
#     }
# }

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


