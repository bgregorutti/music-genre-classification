"""
A minimal dash application for classifying music genre

The layout is based on https://community.plotly.com/t/audio-file-aligned-with-graph/60994
"""

from dash import Dash
from dash.dependencies import Input, Output
import plotly.express as px
from pathlib import Path

from config import environment
from layout import get_layout
from utils import split_data, predict_genre, predict_genre_overall, read_data

# Default sound file
FILE_NAME = Path("../test/resources/mix.wav")

# List the available files
FILE_NAMES = [str(path.name) for path in Path("../test/resources/").glob("*.wav")]

# Get the environment variables
PREDAPP_IP, PREDAPP_PORT = environment()

# Get the data
encoded_sound, sr, df_raw, df = read_data(FILE_NAME)
init_genre_overall = predict_genre_overall(features=split_data(data=df_raw.data.values, sr=sr), host=PREDAPP_IP, port=PREDAPP_PORT)

# Run the application
app = Dash(__name__, title="Audio analysis", update_title=None)
app.logger.error("{}:{}".format(PREDAPP_IP, PREDAPP_PORT))
app.layout = get_layout(file_path=FILE_NAME, file_paths=FILE_NAMES, encoded_sound=encoded_sound, data=init_genre_overall)

# Define the callbacks
@app.callback(
    Output("dropdown_output_container", "children"),
    Output("pred_overall", "data"),
    Output("audiospeler", "src"),
    Input("dropdown_files", "value")
)
def update_dropdown(value):
    global encoded_sound, sr, df_raw, df
    print("Dropdown value", value)
    encoded_sound, sr, df_raw, df = read_data(Path("../test/resources", value))
    encoded_str = "data:audio/mpeg;base64,{}".format(encoded_sound.decode())
    return str(Path("../test/resources", value)), predict_genre_overall(features=split_data(data=df_raw.data.values, sr=sr), host=PREDAPP_IP, port=PREDAPP_PORT), encoded_str

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
    if not features.size:
        prediction = "N/A"
        from_date = -1
        to_date = -1
    else:
        prediction = predict_genre(features=features, host=PREDAPP_IP, port=PREDAPP_PORT)
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
