import asyncio
import threading
import base64
import os
from io import BytesIO

import dash
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc
import dash_daq as daq
from quart import Quart, websocket
from dash_extensions import WebSocket
import dash_bootstrap_components as dbc
import cv2
import numpy as np
from PIL import Image

from files_structure import get_files_structure
from config import config
from style_transfer.style_transfer import StyleTransfer
from text_generation.text_generation import TextGenerator
from deep_dream.deep_dream import DeepDreamer

# since this code is for local use only forgive us those globals
STOP_CAMERA = False
CAPTURE_BUTTON_STATES = {
    True: "NEW IMAGE!",
    False: "CAPTURE IMAGE!"
}
STYLE_TRANSFER = StyleTransfer(config["path_for_hub_models"], out_image_size=1024)
TEXT_GENERATOR = TextGenerator()
DEEP_DREAMER = DeepDreamer()


class VideoCamera(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def load_image(image_path, max_shape=255.):
    raw_image = np.array(Image.open(image_path))
    if len(raw_image.shape) < 3:
        raw_image = cv2.imread(image_path)
    elif raw_image.shape[2] != 3:
        raw_image = np.array(Image.open(image_path).convert("RGB"))
    resize_ratio = min(max_shape / raw_image.shape[0], max_shape / raw_image.shape[1])
    resized_image = Image.fromarray(raw_image).resize(
        (int(resize_ratio * raw_image.shape[1]), int(resize_ratio * raw_image.shape[0]))
    )
    return np.array(resized_image)


server = Quart(__name__)
DELAY_BETWEEN_FRAMES = 0.05  # add delay (in seconds) if CPU usage is too high

STYLE_FILES = get_files_structure(config["styles_dir"], config["file_extension"])["."]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

video_streaming_elem = html.Img(
    id="video",
    style={"width": "100%"}
)

style_transfer_tab = dcc.Tab(
    label="Style transfer",
    value="style_transfer",
    children=html.Div(
        id="style_transfer_body",
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dcc.Dropdown(STYLE_FILES, STYLE_FILES[0], id="style_filename", multi=False),
                    ]),
                    dbc.Row([
                        dbc.Button(
                            "STYLE IT!",
                            id="style_button",
                            color="primary"
                        )
                    ])
                ], width=5),
                dbc.Col([
                    html.Img(id="style_image"),
                    # html.P("Image placeholder")
                ], width=7
                )
            ]),
        ]
    )
)

deep_dream_tab = dcc.Tab(
    label="Deep dream",
    value="deep_dream",
    children=html.Div(
        id="deep_dream_body",
        children=[
            dbc.Row([
                dcc.Dropdown(
                    list(DEEP_DREAMER.models.keys()),
                    list(DEEP_DREAMER.models.keys())[0],
                    id="deep_model_list",
                    multi=False
                ),
                dcc.Dropdown(
                    DEEP_DREAMER.models[list(DEEP_DREAMER.models.keys())[0]]["layers"],
                    DEEP_DREAMER.models[list(DEEP_DREAMER.models.keys())[0]]["layers"][0],
                    id="deep_layers_list",
                    multi=True
                ),
                dcc.Input(id="num_of_steps", type="number", placeholder="Steps per octave (def=15)"),
                dcc.Input(id="step_size", type="number", placeholder="Step size (def=1.)"),
                dcc.Input(id="num_of_octaves", type="number", placeholder="Num of octaves (def=4)"),
                dcc.Input(id="octave_scale", type="number", placeholder="Octave scale (def=1.4)"),
                dbc.Button(
                    "DREAM IT!",
                    id="deep_dream_button",
                    color="primary"
                )
            ])
        ]
    )
)

image_manipulation_tab = dcc.Tab(
    label="Image manipulation",
    value="image_manipulation",
    children=html.Div(
        id="image_manipulation_body",
        children=[
            dbc.Row([
                dbc.Col(
                    [
                        dbc.Row([video_streaming_elem]),
                        dbc.Row([
                            daq.BooleanSwitch(
                                id="capture_image",
                                on=STOP_CAMERA,
                                label=CAPTURE_BUTTON_STATES[STOP_CAMERA],
                                labelPosition="top"
                            )
                        ])
                    ], width=7
                ),
                dbc.Col(html.Div([
                    dcc.Tabs(
                        id="manipulation_tabs",
                        value="style_transfer",
                        children=[
                            style_transfer_tab,
                            deep_dream_tab
                        ]
                    )
                ]), width=5)
            ]),
            dbc.Row([
                html.Div([html.Img(id="transformed_image"), ])
            ]),

        ]
    )
)

text_generator_tab = dcc.Tab(
    label="Text generation",
    value="text_generation",
    children=html.Div(
        id="text_generation_body",
        children=[
            html.H2("GPT-2 - model by OpenAI"),
            html.P("Please provide text input and press 'GENERATE!' button"),
            dcc.Textarea(
                id='text-generator-input',
                value='I was on the Dreamersland music and art festival',
                style={'width': '100%', 'height': 200},
            ),
            dbc.Button(
                "GENERATE!",
                id="generate-text-button",
                color="primary"
            ),
            html.Div(id='text-generator-output', style={'whiteSpace': 'pre-line', 'width': '500px'})
        ]
    )
)

layout = html.Div(
    id="app_body",
    children=[
        dcc.Tabs(
            id="main_tabs",
            value="image_manipulation",
            children=[
                image_manipulation_tab,
                text_generator_tab
            ]
        ),
        WebSocket(url=f"ws://127.0.0.1:5000/stream", id="ws"),
        dcc.Store(id="style_image_data", data=None),
        dcc.Store(id="saved_image", data=None)
    ],
    style={"max_height": "100vh"}
)

app.layout = layout


@server.websocket("/stream")
async def stream():
    global STOP_CAMERA
    camera = VideoCamera(0)
    while 1:
        if DELAY_BETWEEN_FRAMES is not None:
            await asyncio.sleep(DELAY_BETWEEN_FRAMES)
        frame = camera.get_frame()
        if not STOP_CAMERA:
            await websocket.send(f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}")


@app.callback(
    Output("style_image", "src"),
    Output("style_image_data", "data"),
    Input("style_filename", "value")
)
def update_style_image(filename):
    path_to_image = os.path.join(
        config["styles_dir"],
        filename
    )
    ndarray_img = load_image(path_to_image)
    image = Image.fromarray(ndarray_img)
    buffer = BytesIO()
    ext = "png"
    image.save(buffer, format=ext)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{ext};base64, " + encoded, ndarray_img


@app.callback(
    Output("saved_image", "data"),
    Output("capture_image", "label"),
    Input("capture_image", "on"),
    State("ws", "message")
)
def capture_image(n_clicks, raw_image):
    global STOP_CAMERA
    image = raw_image.get("data", None)
    STOP_CAMERA = not STOP_CAMERA
    return np.array(Image.open(BytesIO(base64.b64decode(bytes(image[24:], "UTF-8"))))), CAPTURE_BUTTON_STATES[
        STOP_CAMERA]


app.clientside_callback(  # todo: add photo capturing!
    "function(m){return m ? m.data : '';}",
    Output(f"video", "src"),
    Input(f"ws", "message")
)


@app.callback(
    Output("transformed_image", "src"),
    Input("style_button", "n_clicks"),
    Input("deep_dream_button", "n_clicks"),
    State("saved_image", "data"),
    State("style_image_data", "data"),
    State("num_of_steps", "value"),
    State("step_size", "value"),
    State("num_of_octaves", "value"),
    State("octave_scale", "value"),
    State("deep_model_list", "value"),
    State("deep_layers_list", "value"),
    prevent_initial_call=True
)
def transfer_style(
        style_n_clicks, dream_n_clicks, content_image, style_image, num_of_steps, step_size, num_of_octaves,
        octave_scale, picked_model, layer_values
):
    trigger = dash.callback_context.triggered_id
    if trigger == "style_button":
        preprocessed_image = STYLE_TRANSFER.stylize(content_image, style_image)
    elif trigger == "deep_dream_button":
        if num_of_steps is None:
            num_of_steps = 15
        if step_size is None:
            step_size = 1.
        if num_of_octaves is None:
            num_of_octaves = 4
        if octave_scale is None:
            octave_scale = 1.4
        num_of_steps = int(num_of_steps)
        step_size = float(step_size)
        num_of_octaves = int(num_of_octaves)
        octave_scale = float(octave_scale)
        if not isinstance(layer_values, list):
            layer_values = [layer_values]
        if not len(layer_values):
            return dash.no_update
        layer_names = [DEEP_DREAMER.models[picked_model]["layers"][x] for x in layer_values]
        preprocessed_image = DEEP_DREAMER.perform_deep_dream(
            content_image, picked_model, layer_names, num_of_steps, step_size, num_of_octaves, octave_scale
        )
    image = Image.fromarray(preprocessed_image)
    buffer = BytesIO()
    ext = "png"
    image.save(buffer, format=ext)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{ext};base64, " + encoded


@app.callback(
    Output('text-generator-output', 'children'),
    Input('generate-text-button', 'n_clicks'),
    State('text-generator-input', 'value'),
    prevent_initial_call=True
)
def generate_output(n_clicks, input_text):
    return TEXT_GENERATOR.generate(input_text)


@app.callback(
    Output("deep_layers_list", "options"),
    Output("deep_layers_list", "value"),
    Input("deep_model_list", "value")
)
def pick_model(picked_model):
    layers_for_picked_model = DEEP_DREAMER.models[picked_model]["layers"]
    data = [{"label": layer, "value": layer_idx} for layer_idx, layer in enumerate(layers_for_picked_model)]
    return data, 0


if __name__ == '__main__':
    threading.Thread(target=app.run_server).start()
    server.run()
