import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, Patch
import cv2
import base64
import os
import glob

# Configuration
SESSION_DIR = "/home/sean/humanoid-teleop/scripts/session_20250710_043147"
PLAYBACK_HZ = 0.5

# Load all files
pointcloud_files = sorted(glob.glob(os.path.join(SESSION_DIR, "pointclouds", "*.npy")))
color_files = sorted(glob.glob(os.path.join(SESSION_DIR, "color_images", "*.jpg")))
total_frames = len(pointcloud_files)
print(f"Found {total_frames} frames")

# Create app
app = Dash(__name__)

app.layout = html.Div([
    html.H1(f'Point Cloud Playback @ {PLAYBACK_HZ} Hz', style={'text-align': 'center'}),
    
    # Main display
    html.Div([
        # Point cloud
        dcc.Graph(id='pointcloud', style={'width': '70%', 'height': '600px', 'display': 'inline-block'}),
        # RGB image
        html.Img(id='rgb-image', style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),
    
    # Auto update
    dcc.Interval(id='timer', interval=1000/PLAYBACK_HZ),  # Convert Hz to milliseconds
    dcc.Store(id='frame-counter', data=0),
    dcc.Store(id='initialized', data=False),
])

# Initialize the figure once
@app.callback(
    Output('pointcloud', 'figure'),
    Input('initialized', 'data'),
    prevent_initial_call=False
)
def initialize_figure(initialized):
    if not initialized:
        # Load first frame
        pointcloud_data = np.load(pointcloud_files[0])
        points = pointcloud_data[:, :3]
        colors = pointcloud_data[:, 3:6].astype(int)
        
        # Create initial figure
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=[f'rgb({c[2]}, {c[1]}, {c[0]})' for c in colors],
            ),
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(
                    eye=dict(x=0, y=0, z=-2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=-1, z=0)
                ),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        
        return fig
    
    return Patch()

# Update the data,
@app.callback(
    [Output('pointcloud', 'figure', allow_duplicate=True),
     Output('rgb-image', 'src'),
     Output('frame-counter', 'data'),
     Output('initialized', 'data')],
    [Input('timer', 'n_intervals')],
    [State('frame-counter', 'data')],
    prevent_initial_call=True
)
def update_frame(n_intervals, current_frame):
    if n_intervals is None or n_intervals == 0:
        return Patch(), "", 0, True
    
    # Calculate frame index
    frame_idx = (current_frame + 1) % total_frames
    
    # Load point cloud
    pointcloud_data = np.load(pointcloud_files[frame_idx])
    points = pointcloud_data[:, :3]
    colors = pointcloud_data[:, 3:6].astype(int)
    
    # Load color image
    color_image = cv2.imread(color_files[frame_idx])
    
    # Create patch to update only data
    patched_figure = Patch()
    patched_figure['data'][0]['x'] = points[:, 0]
    patched_figure['data'][0]['y'] = points[:, 1]
    patched_figure['data'][0]['z'] = points[:, 2]
    patched_figure['data'][0]['marker']['color'] = [f'rgb({c[2]}, {c[1]}, {c[0]})' for c in colors]
    
    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', color_image)
    img_str = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
    
    return patched_figure, img_str, frame_idx, True

if __name__ == '__main__':
    print(f"Starting playback at {PLAYBACK_HZ} Hz")
    print("Open http://localhost:8050 in the browser")
    app.run(debug=False, port=8050)