import asyncio
import threading
import base64
import os
from io import BytesIO

import dash
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc
from quart import Quart, websocket
from dash_extensions import WebSocket
import dash_bootstrap_components as dbc
import cv2
import numpy as np
from PIL import Image

from files_structure import get_files_structure
from config import config


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
    resize_ratio = min(max_shape/raw_image.shape[0], max_shape/raw_image.shape[1])
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
                        dcc.Slider(0, 100, 1, value=20, marks=None, id="style_slider")
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
            html.P("Deep dream"),
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
                    html.Div([
                        video_streaming_elem
                    ]), width=7
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
                html.Div([html.P("down")])
            ])

        ]
    )
)

text_generator_tab = dcc.Tab(
    label="Text generation",
    value="text_generation",
    children=html.Div(
        id="text_generation_body",
        children=[
            html.P("Text generation")
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
        dcc.Store(id="style_image_data", data=None)
    ],
    style={"max_height": "100vh"}
)

app.layout = layout


@server.websocket("/stream")
async def stream():
    camera = VideoCamera(0)
    while 1:
        if DELAY_BETWEEN_FRAMES is not None:
            await asyncio.sleep(DELAY_BETWEEN_FRAMES)
        frame = camera.get_frame()
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


app.clientside_callback(  # todo: add photo capturing!
    "function(m){return m ? m.data : '';}",
    Output(f"video", "src"),
    Input(f"ws", "message")
)

if __name__ == '__main__':
    threading.Thread(target=app.run_server).start()
    server.run()
