from dash import dcc, html, dash_table

def get_layout(file_path, file_paths, encoded_sound, data):
    layout = html.Div([
        dcc.Store(id="client_content", data="DATA"),
        dcc.Store(id="client_fig_data", data={}),
        dcc.Store(id="current_position", data=0),
        dcc.Interval(id="client_interval", interval=50),
        dcc.Interval(id="interval_prediction", interval=5000),
        html.H2(children="Music genre classification"),
        dcc.Markdown("A minimal dash application that predict the genre from music data"),
        html.Br(),
        html.Div(
            [
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
        dcc.Graph(id="client_graph"),
        html.Div([
            html.Div([
                html.Div([
                    html.H3(children="Real-time prediction"),
                    html.Div(className="container", children=[dcc.Markdown(id="pred_content", children="N/A")])
                ], className="six columns"),
                html.Div([
                    html.H3(children="Overall prediction"),
                    html.Div(className="datatable_container", children=[
                        dash_table.DataTable(
                            id="pred_overall",
                            columns=[{"name": "Predicted label", "id": "Predicted label", "deletable": False, "selectable": False},
                                    {"name": "Probability", "id": "Probability", "deletable": False, "selectable": False}],
                            data=data
                        )], style={"width": "30%"}),
                ], className="six columns"),
            ], className="row")
        ])
    ])
    return layout