import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import cv2
import base64
import os
import glob

# Configuration
SESSION_DIR = "/home/sean/humanoid-teleop/scripts/pointcloud_recording/session_20250708_213431"
PLAYBACK_HZ = 1

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
])

@app.callback(
    [Output('pointcloud', 'figure'),
     Output('rgb-image', 'src'),
     Output('frame-counter', 'data')],
    [Input('timer', 'n_intervals')],
    [Input('frame-counter', 'data')]
)
def update_frame(n_intervals, current_frame):
    # Calculate frame index
    frame_idx = (current_frame + 1) % total_frames
    
    # Load point cloud
    pointcloud_data = np.load(pointcloud_files[frame_idx])
    points = pointcloud_data[:, :3]
    colors = pointcloud_data[:, 3:6].astype(int)
    
    # Load color image
    color_image = cv2.imread(color_files[frame_idx])
    
    # Create 3D plot
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
    
    # Plot layout
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
    
    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', color_image)
    img_str = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
    
    return fig, img_str, frame_idx

if __name__ == '__main__':
    print(f"Starting playback at {PLAYBACK_HZ} Hz")
    print("Open http://localhost:8050 in the browser")
    app.run(debug=False, port=8050)