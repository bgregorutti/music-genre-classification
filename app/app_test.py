"""
sample code, based on https://community.plotly.com/t/audio-file-aligned-with-graph/60994
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
df["time"] = pd.to_datetime(df["time"], unit="s")


# MAIN --------------------------------------------------------------
app = dash.Dash(__name__, title="Audio analysis", update_title=None)

app.layout = html.Div([
    html.Div(id="client_content", children="DATA"),
    dcc.Store(id="client_fig_data", data={}),
    dcc.Interval(
        id="client_interval",
        interval=50
    ),
    html.Audio(id="audiospeler",
               src="data:audio/mpeg;base64,{}".format(encoded_sound.decode()),
               controls=True,
               autoPlay=False,
               style={"width": "100%"}
               ),
    dcc.Graph(id="client_graph"),
    dcc.Markdown(id="pred_content", children="N/A")

])



# CALLBACKS ---------------------------------------------------------
@app.callback(
    Output("client_fig_data", "data"),
    Input("client_interval", "interval")
)
def update_figure(interval):
    fig = px.line(df, x="time", y="data")
    fig.add_annotation(
        x=0.2,
        y=0.5,
        xref="paper",
        yref="paper",
        xshift=0,
        text="position",
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
            return [{"data": [], "layout": {}}, "ERROR: no data for figure"];
        }
        
        var myaudio = document.getElementById("audiospeler");
        var cur_time = myaudio.currentTime;
        var tot_time = myaudio.duration;
        if( !tot_time ) {
            return [figure_data, "ERROR: no data for sound"];
        }
        /* var ratio_time = cur_time / tot_time */
        var ratio_time = cur_time / tot_time
 
        const fig = Object.assign({}, figure_data, {
            "layout": {
                ...figure_data.layout,
                "annotations": [{
                    ...figure_data.layout.annotations[0],
                    "x": ratio_time,
                    "text": parseFloat(ratio_time).toFixed(2)
                }]
            }
        });       
              
        return [fig, ratio_time];
    }
    """,
    Output("client_graph", "figure"),
    Output("client_content", "children"),
    Input("client_fig_data", "data"),
    Input("client_interval", "n_intervals")
)

if __name__ == "__main__":
    app.run_server(debug=True)