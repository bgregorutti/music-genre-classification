"""
sample code, based on https://community.plotly.com/t/audio-file-aligned-with-graph/60994

v2 improve the figure, dynamically change the line color, WIP
"""

# IMPORTS -----------------------------------------------------------
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly.express as px
import pandas as pd
import base64

import librosa
import numpy as np



# data sound preparations
FILE_NAME = "../test/resources/chicken.wav"
encoded_sound = base64.b64encode(open(FILE_NAME, "rb").read())
np_data, sr = librosa.load(FILE_NAME)
target_sr = 8000
np_data = librosa.resample(np_data, orig_sr=sr, target_sr=target_sr)
time_sec = np.array([t / target_sr for t in range(len(np_data))])

df = pd.DataFrame({
    "time": time_sec,
    "data": np_data
})
df['time'] = pd.to_datetime(df['time'], unit="s")

def initial_figure():
    fig = px.line(df, x="time", y="data")
    fig['data'][0]['line']['color']="lightgray"
    return fig

# MAIN --------------------------------------------------------------
app = dash.Dash(__name__, title="Audio analysis", update_title=None)

app.layout = html.Div([
    html.Div(id='client_content', children='DATA'),
    dcc.Store(id="client_fig_data", data={}),
    dcc.Interval(
        id="client_interval",
        interval=50
    ),
    html.Audio(id='audiospeler',
               src='data:audio/mpeg;base64,{}'.format(encoded_sound.decode()),
               controls=True,
               autoPlay=False,
               style={"width": "100%"}
               ),
    dcc.Graph(id='client_graph', figure=initial_figure()),
    dcc.Markdown(id="pred_content", children="N/A")

])



# CALLBACKS ---------------------------------------------------------
@app.callback(
    Output("client_graph", "figure"),
    [Input("client_interval", "interval"), Input("client_content", "children")]
)
def update_figure(interval, current_position):
    idx = current_position * len(df)
    df["color"] = ["blue"] * (idx - 1) + ["lightgray"] * (len(df) - idx)
    fig = px.line(df, x="time", y="data", color="color")
    fig.add_annotation(
        x=0.2,
        y=0.5,
        xref='paper',
        yref='paper',
        xshift=0,
        text='position',
        showarrow=True,
        font=dict(family="Courier New, monospace", size=14, color="#ffffff"),
        align="center",
        bgcolor="red",
    )
    return fig

@app.callback(
    Output("pred_content", "children"),
    Input("client_content", "children")
)
def update_prediction(client_content):
    return f"Found {client_content}"


# client-side callback with javascript to update graph annotations.
app.clientside_callback(
    """
    function TrackCurrentTime(figure_data, n_intervals){
        if(figure_data === undefined){
            return 'ERROR: no data for figure';
        }
        
        var myaudio = document.getElementById("audiospeler");
        var cur_time = myaudio.currentTime;
        var tot_time = myaudio.duration;
        if( !tot_time ) {
            return 'ERROR: no data for sound';
        }
        var ratio_time = cur_time / tot_time
              
        return ratio_time;
    }
    """,
    Output('client_content', 'children'),
    Input('client_fig_data', 'data'),
    Input('client_interval', 'n_intervals')
)

if __name__ == '__main__':
    app.run_server(debug=True)