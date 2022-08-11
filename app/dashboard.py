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
import math

import librosa
import numpy as np

def split_data(data, sr):
    """
    Split into segments of 3 seconds
    """
    features = []
    num_mfcc = 40
    n_fft = 2048
    hop_length = 512

    total_seconds = data.size / sr
    nr_segments = int(total_seconds / 3)
    sample_per_segment = int(data.size / nr_segments)
    for n in range(nr_segments):
        segment = data[sample_per_segment*n:sample_per_segment*(n+1)]
        mel = librosa.feature.melspectrogram(y=segment, n_mels=40)
        harmonic, percusive = librosa.decompose.hpss(mel)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        spectro = np.stack((mfcc.T, harmonic.T, percusive.T), axis=-1)
        if len(spectro) == math.ceil(sample_per_segment / hop_length):
            features.append(spectro.tolist())
    return np.array(features)

def predict_genre(features):
    from random import choice
    return choice(["rock (51%)", "jazz (95%)", "classical (75%)"])


# data sound preparations
FILE_NAME = "../test/resources/chicken.wav"
encoded_sound = base64.b64encode(open(FILE_NAME, "rb").read())
np_data, sr = librosa.load(FILE_NAME)
time_sec = np.array([t / sr for t in range(len(np_data))])
df_raw = pd.DataFrame({
    "time": np.array([t / sr for t in range(len(np_data))]),
    "data": np_data
})
df_raw["time"] = pd.to_datetime(df_raw["time"], unit="s")

target_sr = 1000
sample = librosa.resample(np_data, orig_sr=sr, target_sr=target_sr)

df = pd.DataFrame({
    "time": np.array([t / target_sr for t in range(len(sample))]),
    "data": sample
})
df["time"] = pd.to_datetime(df["time"], unit="s")





# MAIN --------------------------------------------------------------
app = dash.Dash(__name__, title="Audio analysis", update_title=None)

app.layout = html.Div([
    dcc.Store(id="client_content", data="DATA"),
    dcc.Store(id="client_fig_data", data={}),
    dcc.Interval(id="client_interval", interval=50),
    dcc.Interval(id="client_interval2", interval=5000),
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
    Input("client_content", "data")
)
def update_prediction(client_content):
    position = int(client_content * len(df_raw))
    window = int(3 * sr)
    dat = df_raw.iloc[position: position+window]
    features = split_data(dat.data.values, sr)
    prediction = predict_genre(features)
    print(features.shape)
    return f"Found {client_content}. Current position: {position}. window: {window}. From {dat.time.min()} to {dat.time.max()}\nPrediction: {prediction}"


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
    Output("client_content", "data"),
    Input("client_fig_data", "data"),
    Input("client_interval", "n_intervals")
)

if __name__ == "__main__":
    app.run_server(debug=True)
