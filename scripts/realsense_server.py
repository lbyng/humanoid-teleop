import pyrealsense2 as rs
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import cv2
import base64
from datetime import datetime

# Global variables
pipeline = None
align = None
depth_scale = 0.001
is_running = False

def init_camera():
    """Initialize RealSense camera"""
    global pipeline, align, depth_scale
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        # Create align object
        align = rs.align(rs.stream.color)
        
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return False

def get_frame_data(downsample=4, depth_threshold=3.0):
    """Get current frame data"""
    global pipeline, align, depth_scale
    
    if pipeline is None:
        return None, None, None
    
    try:
        # Get frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get intrinsics
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Generate point cloud
        points = []
        colors = []
        
        h, w = depth_image.shape
        for y in range(0, h, downsample):
            for x in range(0, w, downsample):
                z = depth_image[y, x] * depth_scale
                if z == 0 or z > depth_threshold:
                    continue
                
                # Deproject pixel to 3D point
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], z)
                points.append(point)
                
                # Get color (BGR to RGB)
                color = color_image[y, x]
                colors.append(f'rgb({color[2]}, {color[1]}, {color[0]})')
        
        return np.array(points), colors, cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error getting frame: {e}")
        return None, None, None

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1('RealSense Point Cloud Viewer', style={'text-align': 'center'}),
    
    # Main content
    html.Div([
        # 3D Plot
        html.Div([
            dcc.Graph(id='live-graph', style={'height': '600px'}),
        ], style={'width': '70%', 'display': 'inline-block'}),
        
        # Side panel
        html.Div([
            html.H3('RGB Camera'),
            html.Img(id='live-image', style={'width': '100%', 'max-width': '400px'}),
            html.Div(id='stats', style={'margin-top': '10px'}),
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),
    
    # Controls
    html.Div([
        html.Label('Downsample: '),
        dcc.Slider(id='downsample', min=1, max=8, step=1, value=4,
                   marks={i: str(i) for i in [1, 2, 4, 6, 8]}),
        
        html.Label('Max Depth (m): ', style={'margin-top': '20px'}),
        dcc.Slider(id='max-depth', min=0.5, max=5, step=0.5, value=3,
                   marks={i: f'{i}m' for i in [0.5, 1, 2, 3, 4, 5]}),
    ], style={'padding': '20px', 'width': '60%', 'margin': 'auto'}),
    
    # Update interval
    dcc.Interval(id='graph-update', interval=1000),
])

@app.callback(
    [Output('live-graph', 'figure'),
     Output('live-image', 'src'),
     Output('stats', 'children')],
    [Input('graph-update', 'n_intervals'),
     Input('downsample', 'value'),
     Input('max-depth', 'value')]
)
def update_graph(n, downsample, max_depth):
    """Update graph and image"""
    global is_running
    
    # Initialize camera on first call
    if not is_running:
        if init_camera():
            is_running = True
        else:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="Camera not found", 
                                   xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False)
            return empty_fig, "", "Camera not initialized"
    
    # Get frame data
    points, colors, rgb_image = get_frame_data(downsample, max_depth)
    
    # Create figure
    if points is not None and len(points) > 0:
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
            ),
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',

                camera=dict(
                    eye=dict(x=0, y=0, z=-2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=-1, z=0)
                ),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        
        stats_text = f"Points: {len(points):,}"
    else:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        stats_text = "No points"
    
    # Convert image to base64
    img_str = ""
    if rgb_image is not None:
        _, buffer = cv2.imencode('.jpg', rgb_image)
        img_str = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
    
    return fig, img_str, stats_text

if __name__ == '__main__':
    # Check for RealSense devices
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("No RealSense device found!")
    else:
        print("RealSense device detected")
        print("Starting server at http://localhost:8050")
        app.run(debug=False, host='0.0.0.0', port=8050)