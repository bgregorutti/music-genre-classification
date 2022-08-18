"""
sample code, based on https://community.plotly.com/t/audio-file-aligned-with-graph/60994
"""

#TODO add the overall prediction, based on the all music, in a separated Markdown

# IMPORTS -----------------------------------------------------------
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly.express as px
import pandas as pd
import base64
import math
import requests
from pathlib import Path

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
    if not nr_segments:
        nr_segments = 1
    
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
    response = requests.post("http://predapp:8080/classify", json=features.tolist()) #TODO env file
    predicted_label = response.json().get("predicted_label")
    probability = round(response.json().get("probability") * 100, 2)
    return f"{predicted_label} ({probability}%)"

def predict_genre_overall(features):
    response = requests.post("http://predapp:8080/classify_overall", json=features.tolist()) #TODO env file
    if response.status_code != 200:
        return "N/A"

    predicted_label = response.json().get("predicted_label")
    probability = round(response.json().get("probability") * 100, 2)
    return f"{predicted_label} ({probability}%)"

# Default sound file
FILE_NAME = Path("../test/resources/javanaise.wav")

# List the available files
FILE_NAMES = [str(path.name) for path in Path("../test/resources/").glob("*.wav")]

def read_data(file_name):
    # Open the default file and process it
    encoded_sound = base64.b64encode(open(str(file_name), "rb").read())
    np_data, sr = librosa.load(str(file_name))
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

    return encoded_sound, np_data, sr, df_raw, df

encoded_sound, np_data, sr, df_raw, df = read_data(FILE_NAME)

# MAIN --------------------------------------------------------------
app = dash.Dash(__name__, title="Audio analysis", update_title=None)

app.layout = html.Div([
    dcc.Store(id="client_content", data="DATA"),
    dcc.Store(id="client_fig_data", data={}),
    dcc.Store(id="current_position", data=0),
    dcc.Interval(id="client_interval", interval=50),
    dcc.Interval(id="interval_prediction", interval=5000),
    html.H2(children="Music genre classification"),
    html.Br(),
    html.Div(
        [
            dcc.Dropdown(FILE_NAMES, str(FILE_NAME.name), id="dropdown_files", style={"width": "40%"}),
            html.Div(id="dropdown_output_container")
        ]
    ),
    html.Audio(id="audiospeler",
               src="data:audio/mpeg;base64,{}".format(encoded_sound.decode()),
               controls=True,
               autoPlay=False,
               style={"width": "100%"}),
    dcc.Graph(id="client_graph"),
    html.H2(children="Real-time prediction"),
    html.Div(className="container", children=[dcc.Markdown(id="pred_content", children="N/A")]),
    html.H2(children="Overall prediction"),
    dcc.Markdown(id="pred_overall", children=predict_genre_overall(split_data(data=np_data, sr=sr)))
])



# CALLBACKS ---------------------------------------------------------

@app.callback(
    Output("dropdown_output_container", "children"),
    Output("pred_overall", "children"),
    Output("audiospeler", "src"),
    Input("dropdown_files", "value")
)
def update_dropdown(value):
    global encoded_sound, np_data, sr, df_raw, df
    print("Dropdown value", value)
    encoded_sound, np_data, sr, df_raw, df = read_data(Path("../test/resources", value))
    encoded_str = "data:audio/mpeg;base64,{}".format(encoded_sound.decode())
    return str(Path("../test/resources", value)), predict_genre_overall(split_data(data=np_data, sr=sr)), encoded_str

@app.callback(
    Output("client_fig_data", "data"),
    Input("client_interval", "interval")
)
def update_figure(interval):
    return waveplot(df)

def waveplot(df):
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
    Input("current_position", "data")
)
def update_prediction(current_position):
    # app.logger.info("Updating the prediction")
    position = int(current_position * len(df_raw))
    window = int(3 * sr)
    dat = df_raw.iloc[position:position+window]
    features = split_data(dat.data.values, sr)
    prediction = predict_genre(features)
    from_date = f"00:{dat.time.min().minute:02d}:{dat.time.min().second:02d}"
    to_date = f"00:{dat.time.max().minute:02d}:{dat.time.max().second:02d}"
    return f"""
**Current position**: {position}

**From** {from_date} **to** {to_date}

**Prediction**: {prediction}
"""


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

app.clientside_callback(
    """
    function GetCurrentPosition(figure_data, n_intervals){
        if(figure_data === undefined){
            return "ERROR: no data for figure";
        }
        
        var myaudio = document.getElementById("audiospeler");
        var cur_time = myaudio.currentTime;
        var tot_time = myaudio.duration;
        if( !tot_time ) {
            return "ERROR: no data for sound";
        }
        var ratio_time = cur_time / tot_time   
              
        return ratio_time;
    }
    """,
    Output("current_position", "data"),
    Input("client_fig_data", "data"),
    Input("interval_prediction", "n_intervals")
)

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
