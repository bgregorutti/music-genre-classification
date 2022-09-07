from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

def get_sidebar_style():
    """
    Returns the sidebar style
    """
    return {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "32rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

def get_content_style():
    """
    Returns the content style
    """
    return {
        "margin-left": "36rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

def get_layout(file_path, file_paths, encoded_sound, data):

    sidebar_layout = html.Div(style=get_sidebar_style(), children=[
        html.H2(children="Music genre classification", className="display-4"),
        dcc.Markdown("A minimal dash application that predicts the genre from music data", className="lead"),
        html.Hr(),
        html.H3(children="Real-time prediction", className="display-6"),
        html.Div(className="container", children=[dcc.Markdown(id="pred_content", children="N/A", className="lead")]),
        html.Br(),
        html.H3(children="Overall prediction", className="display-6"),
        html.Div(className="datatable_container", children=[
            dash_table.DataTable(
                id="pred_overall",
                columns=[{"name": "Predicted label", "id": "Predicted label", "deletable": False, "selectable": False},
                        {"name": "Probability", "id": "Probability", "deletable": False, "selectable": False}],
                data=data)
        ]),
        html.Br(),
        html.Br()
    ])

    content_layout = html.Div(style=get_content_style(), children=[
        html.Br(),
        html.Div(
            [
                dcc.Markdown("Choose the music you want for testing:", className="lead"),
                dcc.Dropdown(file_paths, str(file_path.name), id="dropdown_files", style={"width": "40%"}),
                html.Div(id="dropdown_output_container")
            ]
        ),
        html.Br(),
        html.Br(),
        html.Audio(id="audiospeler",
                   src="data:audio/mpeg;base64,{}".format(encoded_sound.decode()),
                   controls=True,
                   autoPlay=False,
                   style={"width": "100%"}),
        dcc.Graph(id="client_graph")
    ])

    layout = html.Div([
        dcc.Store(id="client_content", data="DATA"),
        dcc.Store(id="client_fig_data", data={}),
        dcc.Store(id="current_position", data=0),
        dcc.Interval(id="client_interval", interval=50),
        dcc.Interval(id="interval_prediction", interval=5000),
        sidebar_layout,
        content_layout
    ])

    return layout