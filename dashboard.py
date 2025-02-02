from hospital_base import *
import json
import pandas as pd
from tqdm import tqdm

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
        self.dist = Distribiutions(cnfg=None)
        Section.dist = self.dist
        self.hospital = Hospital()
        self.client_generator = ClientGeneratorForHospital(
            targeted_hospital=self.hospital,
            dist=self.dist,
            capacity= SECTION_CAPACITIES.get(SectionType.OUTSIDE)
        )
        self.nature = Nature(targeted_hospital=self.hospital, dist=self.dist)
        # self.state = simulation_state
        self.tasks: List[asyncio.Task] = []

    async def start(self):
        # Start hospital sections
        self.tasks.append(asyncio.create_task(self.hospital.emergency.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.pre_surgery.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.labratory.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.operating_rooms.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.ward.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.icu.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.hospital.ccu.run(self.dist)))
        
        # Start client generator
        self.tasks.append(asyncio.create_task(self.client_generator.run_patient_generator()))
        # self.tasks.append(asyncio.create_task(self.client_generator.run(self.dist)))
        self.tasks.append(asyncio.create_task(self.nature.run()))
        # self.tasks.append(asyncio.create_task(self.nature.rip.run(self.dist)))

        # Start state updater
        self.tasks.append(asyncio.create_task(self.update_state()))
        print("[Simulation] Started.")

    async def update_state(self):
        global SIMULATION_CLOCK
        update_interval = 1  # seconds (real time)
        power_outage_day = self.dist.uniform_dist(1, 30)
        # power_outage_day = 1
        outage = False
        while True:
            await asyncio.sleep(update_interval)  # Wait for the update interval
            # Increment simulation clock based on speed and interval
            SIMULATION_CLOCK += update_interval * SIMULATION_SPEED
            days, remainder = divmod(int(SIMULATION_CLOCK), 86400)  # 86400 seconds in a day
            day_in_this_month = days % 30
            print(f"[Nature] Next Power outage is on day {power_outage_day} / now is day {day_in_this_month}")
            if day_in_this_month == power_outage_day and not outage:
                print("power off!")
                await self.hospital.electricity_suspension()
                start_outage_time = time.time()
                outage = True
            if outage is True and (time.time() - start_outage_time) >= 24:
                print("power on!")
                await self.hospital.icu.restore_capacity(reduction_percent=REDUCTION_PERCENT)
                await self.hospital.ccu.restore_capacity(reduction_percent=REDUCTION_PERCENT)
                outage = False

    async def stop(self):
        print("[Simulation] Stopping...")
        await self.client_generator.stop_all()
        await self.hospital.emergency.stop()
        await self.hospital.pre_surgery.stop()
        await self.hospital.labratory.stop()
        await self.hospital.ccu.stop()
        await self.hospital.icu.stop()
        await self.hospital.operating_rooms.stop()
        await self.hospital.ward.stop()
        await self.nature.rip.stop()
        await self.nature.stop()
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



import dash
from dash import dcc, html, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import asyncio
import pandas as pd
import threading
import logging
from dash.dependencies import MATCH, ALL, ALLSMALLER
import json  # For safer JSON parsing
import dash_table

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the simulation
simulation = Simulation()

# Define the fixed sidebar style (Commented Out)
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "250px",
    "padding": "20px",
    "background-color": "#34495e",
    "overflow": "auto",
}

# Define the main content style
CONTENT_STYLE = {
    "margin-left": "20px",  # Adjusted to prevent overlap since sidebar is commented out
    "margin-right": "20px",
    "padding": "20px",
}

def get_active_sections():
    return [section_type for section_type in SectionType if section_type not in [SectionType.OUTSIDE, SectionType.RIP]]

def generate_section_card(section_type):
    """
    Generates a Bootstrap card for a given section with nested tabs:
    - Outer Tabs: "Patients" and "Worker Assignments"
    - Inner Tabs within "Worker Assignments": One tab per worker
    """
    return dbc.Card(
        [
            # Section Title
            dbc.CardHeader(
                html.H4(f"{section_type.value} Section", className="text-center text-primary")
            ),
            
            # Outer Tabs
            dbc.CardBody(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Patients", tab_id="patients"),
                            dbc.Tab(label="Worker Assignments", tab_id="workers"),
                        ],
                        id=f'tabs-outer-{section_type.value.lower()}',
                        active_tab="patients",
                    ),
                    html.Div(id=f'content-outer-{section_type.value.lower()}')
                ]
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

def generate_event_table():
    # Assuming `Section.all_events` is a global list of Event instances
    event_data = []
    for event in Section.all_events:
        event_data.append({
            "Event Type": event.event_type.value,
            "Event Time": event.event_time,
            "Patient ID": event.event_patient.id if event.event_patient else "N/A",
            "Patient Type": event.event_patient.patient_type.value if event.event_patient else "N/A",
            "Surgery Type": event.event_patient.surgery_type.value if event.event_patient else "N/A",
        })
    
    if not event_data:
        return dbc.Alert("No events to display.", color="info", className="text-center")
    
    df = pd.DataFrame(event_data[::-1])  # Reverse to show latest events first
    
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=10,  # Number of rows per page
        style_table={'overflowX': 'auto'},
        style_cell={
            'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
            'whiteSpace': 'normal',
            'textAlign': 'center',
            'backgroundColor': '#1c1c1c',
            'color': 'white'
        },
        style_header={
            'backgroundColor': '#2980b9',
            'fontWeight': 'bold',
            'color': 'white'
        },
        style_data={
            'backgroundColor': '#1c1c1c',
            'color': 'white'
        },
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action='native',
    )
    
    return table

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
        tab_id = {'type': 'worker-tab', 'section': section_type.value.lower(), 'worker': worker_id}
        tabs.append(dbc.Tab(label=f"Worker {worker_id}", tab_id=tab_id))
    
    return dbc.Tabs(
        tabs,
        id={'type': 'tabs-inner', 'section': section_type.value.lower()},
        active_tab={'type': 'worker-tab', 'section': section_type.value.lower(), 'worker': list(workers)[0]} if workers else None,
    )

def generate_dynamic_plot():
    """
    Generates the Live Dynamic Plot with multiple charts stacked vertically.
    """
    # Define the number of dynamic charts you want
    num_charts = 2* len(get_active_sections())  # You can adjust this number as needed
    num_charts = 14
    dynamic_charts = []
    for i in range(1, num_charts + 1):
        dynamic_charts.append(
            dbc.Card(
                dbc.CardBody([
                    html.H4(f"Live Dynamic Plot {i}", className="text-center text-primary"),
                    dcc.Graph(
                        id=f'dynamic-plot-{i}',
                        figure={
                            'data': [
                                go.Scatter(
                                    x=[],
                                    y=[],
                                    mode='lines',
                                    name=f'Variable {i}'
                                )
                            ],
                            'layout': go.Layout(
                                title=f'Dynamic Variable {i} Over Time',
                                xaxis={'title': 'Time'},
                                yaxis={'title': 'Value'},
                                plot_bgcolor='#2c3e50',
                                paper_bgcolor='#2c3e50',
                                font={'color': 'white'}
                            )
                        },
                        config={'displayModeBar': False}
                    )
                ]),
                style={
                    'width': '100%',
                    'margin': '10px 0',
                    'border': '2px solid #7f8c8d',
                    'borderRadius': '10px',
                    'backgroundColor': '#2c3e50'
                }
            )
        )
    
    return html.Div(dynamic_charts)

# Include Font Awesome in the external_stylesheets
external_stylesheets = [
    dbc.themes.CYBORG,  # Existing Bootstrap theme
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"  # Font Awesome CDN
]

# Define Dash layout using Dash Bootstrap Components
app_dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app_dash.title = "Hospital Simulation Dashboard"

app_dash.layout = html.Div([
    # Sidebar (Control Panels) - Currently Removed (Commented Out)
    # html.Div([
    #     html.H2("Control Panels", className="text-center text-light"),
    #     dbc.Button("Pause All", id='pause-all', color="danger", className="mb-2 btn-block", n_clicks=0),
    #     dbc.Button("Resume All", id='resume-all', color="success", className="mb-2 btn-block", n_clicks=0, disabled=True),
    #     # Individual section controls within an accordion
    #     dbc.Accordion([
    #         dbc.AccordionItem([
    #             dbc.Button("Pause PRE_SURGERY", id='pause-pre_surgery', color="danger", className="mb-2 btn-block", n_clicks=0),
    #             dbc.Button("Resume PRE_SURGERY", id='resume-pre_surgery', color="success", className="mb-2 btn-block", n_clicks=0, disabled=True),
    #         ], title="PRE_SURGERY"),
    #         dbc.AccordionItem([
    #             dbc.Button("Pause EMERGENCY", id='pause-emergency', color="danger", className="mb-2 btn-block", n_clicks=0),
    #             dbc.Button("Resume EMERGENCY", id='resume-emergency', color="success", className="mb-2 btn-block", n_clicks=0, disabled=True),
    #         ], title="EMERGENCY"),
    #         # Add more AccordionItems for other sections
    #     ], start_collapsed=True),
    # ], style=SIDEBAR_STYLE),
    
    # Main Content Area
    html.Div([
        # Header with Event Table and Live Plot Toggle Buttons
        html.Div([
            dbc.Button(
                [html.I(className="fa fa-table fa-lg me-2"), "Event Table"],  # Using html.I for Font Awesome icon
                id='toggle-event-table',
                color="secondary",
                className="me-3",  # Margin-end for spacing
                n_clicks=0
            ),
            dbc.Button(
                [html.I(className="fa fa-chart-line fa-lg me-2"), "Live Plot"],  # Live Plot Button
                id='toggle-live-plot',
                color="primary",
                className="",  # No additional class needed
                n_clicks=0
            ),
        ], style={"textAlign": "left", "marginBottom": "20px"}),  # Align buttons to the left and add spacing
        
        html.H1("Hospital Simulation Dashboard", className="text-center text-primary mb-4"),
        
        # Conditional Content: Dashboard, Event Table, or Live Plot
        html.Div(id='main-content')
    ], style=CONTENT_STYLE),
    
    # Simulation Clock (fixed position)
    html.Div(
        dbc.Alert(id='clock-display', color="info", className="text-center"),
        style={
            'position': 'fixed',
            'top': '20px',
            'right': '20px',  # Positioned to the right
            'zIndex': '9999',
            'width': '200px'
        }
    ),
    
    # Interval Component for Callbacks (Placed outside the grid)
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # 1 second in milliseconds
        n_intervals=0
    )
])

# Callback to toggle between Dashboard, Event Table, and Live Plot
@app_dash.callback(
    Output('main-content', 'children'),
    [Input('toggle-event-table', 'n_clicks'),
     Input('toggle-live-plot', 'n_clicks')],
    [State('toggle-event-table', 'n_clicks'),
     State('toggle-live-plot', 'n_clicks')]
)
def toggle_main_content(n_clicks_table, n_clicks_plot, state_clicks_table, state_clicks_plot):
    # Determine which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'toggle-event-table':
        # Show Event Table
        return dbc.Card(
            dbc.CardBody([
                dcc.Loading(
                    id="loading-event-table",
                    type="circle",
                    children=generate_event_table()
                )
            ]),
            style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px'}
        )
    elif button_id == 'toggle-live-plot':
        # Show Live Plot with Multiple Charts
        return dbc.Card(
            dbc.CardBody([
                dcc.Loading(
                    id="loading-dynamic-plots",
                    type="circle",
                    children=generate_dynamic_plot()
                )
            ]),
            style={'backgroundColor': '#2c3e50', 'padding': '20px', 'borderRadius': '10px'}
        )
    else:
        # Show Dashboard Content with Loading Indicators
        return dbc.Card(
            dbc.CardBody([
                dcc.Loading(
                    id="loading-live-status",
                    type="circle",
                    children=dbc.Card(
                        dbc.CardBody([
                            html.H3("Live Status", className="text-center text-light"),
                            html.Div(id='live-status-table')
                        ]),
                        className="mb-4",
                        style={'backgroundColor': '#34495e', 'border': 'none'}
                    )
                ),
                
                dcc.Loading(
                    id="loading-patient-visualization",
                    type="circle",
                    children=dbc.Card(
                        dbc.CardBody([
                            html.H3("Patient Visualization", className="text-center text-light"),
                            dbc.Row([
                                dbc.Col(
                                    generate_section_card(section_type),
                                    width=12,
                                    md=6
                                ) for section_type in get_active_sections()
                            ])
                        ]),
                        className="mb-4",
                        style={'backgroundColor': '#34495e', 'border': 'none'}
                    )
                ),
            ])
        )

# Callback to update live status table with section capacity
@app_dash.callback(
    Output('live-status-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_live_status_table(n_intervals):
    active_sections = get_active_sections()
    table_header = [
        html.Thead(
            html.Tr([
                html.Th("Section", className="text-center"),
                html.Th("Entities", className="text-center"),
                html.Th("Queue Length", className="text-center")
            ], style={'backgroundColor': '#2980b9', 'color': 'white'})
        )
    ]
    table_body = []
    for section_type in active_sections:
        section = simulation.hospital.get_section(section_type)
        if section:
            entities_count = len(section.entities)
            queue_count = section.queue.qsize()
            capacity = section.capacity
            # n_nonelective, n_elective= n_nonelective_n_elective(section.queue_lock)
            section_display = f"{section_type.value} (Servers/Queue: {capacity.servers}/{capacity.queue})"
            row = html.Tr([
                html.Td(section_display, className="text-center"),
                html.Td(str(entities_count), className="text-center"),
                html.Td(f"{queue_count}", className="text-center")
            ])
        else:
            row = html.Tr([
                html.Td(f"{section_type.value} (Capacity: N/A)", className="text-center"),
                html.Td("0", className="text-center"),
                html.Td("0", className="text-center")
            ])
        table_body.append(row)
    table = dbc.Table(
        table_header + [html.Tbody(table_body)],
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        striped=True
    )
    return table

# Callback to update patient visualization graphs
@app_dash.callback(
    [Output(f'graph-{section_type.value.lower()}', 'figure') for section_type in get_active_sections()],
    Input('interval-component', 'n_intervals')
)
def update_graphs(n_intervals):
    figures = []
    MAX_DISPLAY = 20  # Maximum number of patients to display per section
    active_sections = get_active_sections()
    for section_type in active_sections:
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
        Output(f'content-outer-{section_type.value.lower()}', 'children') for section_type in get_active_sections()
    ],
    [
        Input(f'tabs-outer-{section_type.value.lower()}', 'active_tab') for section_type in get_active_sections()
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
    active_sections = get_active_sections()
    # print("checking")
    # print(active_sections, len(active_sections))
    # print(active_tabs_outer)
    # print(args)
    # print(30*"^")
    
    # Generate patient graphs first
    figures = update_graphs(n_intervals)
    
    for idx, section_type in enumerate(active_sections):
        active_tab = active_tabs_outer[idx]
        if active_tab == "patients":
            # Display the patient graph
            figures = update_graphs(n_intervals)
            graph_idx = active_sections.index(section_type)
            contents.append(
                dcc.Graph(
                    id=f'graph-inner-{section_type.value.lower()}',
                    figure=figures[graph_idx],
                    config={'displayModeBar': False},
                    style={
                        'height': '400px',  # Increased height for better visibility
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
                        # Dynamically generate worker content areas with pattern-matching IDs
                        *[
                            html.Div(
                                id={'type': 'worker-content', 'section': section_type.value.lower(), 'worker': worker_id}
                            ) 
                            for worker_id in simulation.hospital.get_section(section_type).worker_to_patient.keys()
                        ]
                    ])
                )
        else:
            contents.append(html.P("Select a tab to view content."))
    
    return contents

# Pattern-Matching Callback to update worker contents
@app_dash.callback(
    Output({'type': 'worker-content', 'section': MATCH, 'worker': MATCH}, 'children'),
    Input({'type': 'worker-tab', 'section': MATCH, 'worker': MATCH}, 'active_tab')
)
def update_worker_content(active_tab):
    """
    Updates the content within each worker's inner tab.
    Displays the patient the worker is currently serving.
    """
    ctx = callback_context
    if not ctx.triggered:
        return ""
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        triggered_id = json.loads(triggered_id.replace("'", "\""))  # Safer parsing
    except json.JSONDecodeError:
        print(f"Failed to parse triggered_id: {triggered_id}")
        return dbc.Alert("Invalid triggered ID.", color="danger")
    
    print("Triggered ID:", triggered_id)  # Debugging Statement
    
    section = triggered_id['section'].upper()
    worker_id = triggered_id['worker']
    
    # Convert section string back to SectionType enum
    try:
        section_type = SectionType(section)
    except ValueError:
        print(f"Invalid section: {section}")
        return dbc.Alert("Invalid section.", color="danger")
    
    section_obj = simulation.hospital.get_section(section_type)
    if not section_obj:
        print(f"Section object not found for: {section_type}")
        return dbc.Alert("Section not found.", color="danger")
    
    patient = section_obj.worker_to_patient.get(worker_id)
    
    if patient:
        print(f"Worker {worker_id} is assigned to Patient {patient.id}")  # Debugging Statement
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
        print(f"Worker {worker_id} has no assigned patient.")  # Debugging Statement
        return dbc.Alert("No patient assigned.", color="info")

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

# Callback to handle pause and resume buttons (if Control Panels are re-enabled)
@app_dash.callback(
    [
        Output(f'pause-{section_type.value.lower()}', 'disabled') for section_type in get_active_sections()
    ] +
    [
        Output(f'resume-{section_type.value.lower()}', 'disabled') for section_type in get_active_sections()
    ],
    [
        Input(f'pause-{section_type.value.lower()}', 'n_clicks') for section_type in get_active_sections()
    ] +
    [
        Input(f'resume-{section_type.value.lower()}', 'n_clicks') for section_type in get_active_sections()
    ]
)
def update_button_states(*args):
    num_sections = len(get_active_sections())
    pause_clicks = args[:num_sections]
    resume_clicks = args[num_sections:]
    
    # Initialize lists for button states
    pause_disabled = []
    resume_disabled = []
    
    for idx, section_type in enumerate(get_active_sections()):
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
        Input(f'pause-{section_type.value.lower()}', 'n_clicks') for section_type in get_active_sections()
    ] +
    [
        Input(f'resume-{section_type.value.lower()}', 'n_clicks') for section_type in get_active_sections()
    ]
)
def control_sections(*args):
    num_sections = len(get_active_sections())
    pause_clicks = args[:num_sections]
    resume_clicks = args[num_sections:]
    
    for idx, section_type in enumerate(get_active_sections()):
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

# Callback to update each Dynamic Plot
@app_dash.callback(
    [
        Output(f'dynamic-plot-{i}', 'figure') for i in range(1, 2* len(get_active_sections())+1)
    ],
    Input('interval-component', 'n_intervals'))
def update_dynamic_plots(n_intervals):
    # Example: Creating three different dynamic plots
    import math
    figures = []
    for i in get_active_sections():
        section = simulation.hospital.get_section(i)
        x_q, y_q = section.queue_size_time_series
        x_e, y_e = section.entity_size_time_series
        # y = [math.sin((i * j + n_intervals) * 0.1) for j in x]  # Different sine waves
        moving_avg_q = pd.Series(y_q).rolling(window=10, min_periods=1).mean()
        moving_avg_e = pd.Series(y_e).rolling(window=10, min_periods=1).mean()

        figure = {
            'data': [
                go.Scatter(
                    x=x_q,
                    y=y_q,
                    mode='lines',
                    name=f'{i.value} queue size',
                    line=dict(color='cyan')
                ),
                go.Scatter(x=x_q,
                    y=moving_avg_q,
                    mode='lines',
                    name='Moving Average',
                    line=dict(color='yellow', dash='dash')

                ) 
            ],
            'layout': go.Layout(
                title=f'{i.value} queue size Over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Value'},
                plot_bgcolor='#2c3e50',
                paper_bgcolor='#2c3e50',
                font={'color': 'white'}
            )
        }
        figures.append(figure)
        figure = {
            'data': [
                go.Scatter(
                    x=x_e,
                    y=y_e,
                    mode='lines',
                    name=f'{i.value} entity size',
                    line=dict(color='cyan')
                ),
                go.Scatter(x=x_e,
                    y=moving_avg_e,
                    mode='lines',
                    name='Moving Average',
                    line=dict(color='yellow', dash='dash')

                ) 
                            
            ],
            'layout': go.Layout(
                title=f'{i.value} entity size Over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Value'},
                plot_bgcolor='#2c3e50',
                paper_bgcolor='#2c3e50',
                font={'color': 'white'}
            )
        }
        figures.append(figure)
    return figures

# Function to run Dash in a separate thread
def run_dash():
    app_dash.run_server(debug=False, use_reloader=False, port=8060)

# Main Async Function
async def main():
    # Start Dash in a separate thread
    dash_thread = threading.Thread(target=run_dash, daemon=True)
    dash_thread.start()

    # Start the simulation
    simulation_task = asyncio.create_task(simulation.run_simulation(duration=750))

    # Wait for the simulation to finish
    await simulation_task




def save_things(i: int):
    path = f'result_A_{i}/'
    # path = f'result{i}/'
    event_data = []
    for event in tqdm(Section.all_events):
        event_data.append({
            "Event Type": event.event_type.value,
            "Event Time": event.event_time,
            "Patient ID": event.event_patient.id if event.event_patient else "N/A",
            "Patient Type": event.event_patient.patient_type.value if event.event_patient else "N/A",
            "Surgery Type": event.event_patient.surgery_type.value if event.event_patient else "N/A",
        })
    
    pd.DataFrame(event_data).to_excel(path+'events.xlsx')


    # 
    patient_data = []
    for p in tqdm(Section.all_patients):
        total_serving_duration = 0
        total_queue_duration = 0

        dict_p = {
                "Patient id": p.id,
                "Patient Type": p.patient_type.value,
                "Surgery Type": p.surgery_type.value,
                "re-surgery times": p.re_surgery_times
                
            }
        for sec in tqdm(get_active_sections()):

            # if sec == SectionType.WARD:
                # print(p.section_entry_leave_time)
            section_entry_leave_time = p.section_entry_leave_time.get(sec)
            queue_entry_leave_time = p.queue_entry_leave_time.get(sec)
            # if not section_entry_leave_time or not queue_entry_leave_time:
            #     dict_p[f"{sec.value} Serving Duration"] = None
            #     dict_p[f"{sec.value} Queue Duration"] = None
            #     continue
            # print(sec.value)
            try:
                s_duration = section_entry_leave_time[1] - section_entry_leave_time[0]
                if sec in [SectionType.EMERGENCY, SectionType.PRE_SURGERY]:
                    s_duration += (section_entry_leave_time[3] - section_entry_leave_time[2])
                q_duration = queue_entry_leave_time[1] - queue_entry_leave_time[0]
                dict_p[f"{sec.value} Serving Duration"] = s_duration
                dict_p[f"{sec.value} Queue Duration"] = q_duration
                total_queue_duration += q_duration
                total_serving_duration += s_duration
            except TypeError:
                dict_p[f"{sec.value} Serving Duration"] = None
                dict_p[f"{sec.value} Queue Duration"] = None
        dict_p["total_serving_duration"] = total_serving_duration
        dict_p["total_queue_duration"] = total_queue_duration
        dict_p["total_time_in_system"] = total_queue_duration + total_serving_duration

        patient_data.append(dict_p)
        pd.DataFrame(patient_data).to_excel(path+'patients.xlsx')


    for sec in tqdm(get_active_sections()):
        instance_sec = Section.__instances__.get(sec.value)
        x,y = instance_sec.entity_size_time_series
        x2, y2 = instance_sec.queue_size_time_series

        pd.DataFrame(y2, index=x2).to_excel(path + f'{sec.value} queue.xlsx')
        pd.DataFrame(y, index=x).to_excel(path + f'{sec.value} entity.xlsx')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    
    finally:
        save_things(1)
