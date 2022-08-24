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

# Run the application
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, title="Audio analysis", update_title=None, external_stylesheets=external_stylesheets)

# Path of the assets
RESOURCE_FOLDER = Path("assets/")

# Path of the default WAV file
FILE_NAME = Path(RESOURCE_FOLDER, "mix.wav")

# List the available files
FILE_NAMES = [str(path.name) for path in Path(RESOURCE_FOLDER).glob("*.wav")]

# Get the environment variables
PREDAPP_IP, PREDAPP_PORT = environment()

# Get the data
ENCODED_SOUND, SAMPLE_RATE, RAW_DATA, df = read_data(FILE_NAME)
INIT_GENRE = predict_genre_overall(features=split_data(data=RAW_DATA.data.values, sr=SAMPLE_RATE), host=PREDAPP_IP, port=PREDAPP_PORT)

app.logger.error("{}:{}".format(PREDAPP_IP, PREDAPP_PORT))
app.layout = get_layout(file_path=FILE_NAME, file_paths=FILE_NAMES, encoded_sound=ENCODED_SOUND, data=INIT_GENRE)

# Define the callbacks
@app.callback(
    Output("dropdown_output_container", "children"),
    Output("pred_overall", "data"),
    Output("audiospeler", "src"),
    Input("dropdown_files", "value")
)
def update_dropdown(value):
    """
    Callback for the dropdown

    Args:
        value: value of the dropdown
    
    Returns:
        Dropdown output, a prediction of the music genre and the encoded sound
    """
    global ENCODED_SOUND, SAMPLE_RATE, RAW_DATA, SAMPLE_DATA
    print("Dropdown value", value)
    ENCODED_SOUND, SAMPLE_RATE, RAW_DATA, SAMPLE_DATA = read_data(Path(RESOURCE_FOLDER, value))
    encoded_str = "data:audio/mpeg;base64,{}".format(ENCODED_SOUND.decode())
    dropdown_output = f"File loaded: {Path(RESOURCE_FOLDER, value)}"
    return dropdown_output, predict_genre_overall(features=split_data(data=RAW_DATA.data.values, sr=SAMPLE_RATE), host=PREDAPP_IP, port=PREDAPP_PORT), encoded_str

@app.callback(
    Output("client_fig_data", "data"),
    Input("client_interval", "interval")
)
def update_figure(interval):
    return waveplot(SAMPLE_DATA)

def waveplot(df):
    """
    Plot the signal

    Args:
        df: a DataFrame object with columns ["time", "data"]
    
    Returns:
        A plotly's Figure object
    """
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
    """
    Update the prediction wrt the current song position
    """
    position = int(current_position * len(RAW_DATA))
    window = int(3 * SAMPLE_RATE)
    dat = RAW_DATA.iloc[position:position+window]
    features = split_data(dat.data.values, SAMPLE_RATE)
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

# client-side callbacks with javascript to update graph annotations and position
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
