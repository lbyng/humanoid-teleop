import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
import json

class RealSenseRecorder:
    def __init__(self, output_dir="pointcloud_recording"):
        self.output_dir = output_dir
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.frame_count = 0
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Subdirectories
        self.pointcloud_dir = os.path.join(self.session_dir, "pointclouds")
        self.color_dir = os.path.join(self.session_dir, "color_images")
        self.depth_dir = os.path.join(self.session_dir, "depth_images")
        os.makedirs(self.pointcloud_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        self.metadata = {
            "session_start": timestamp,
            "frames": []
        }
        
    def start_camera(self):
        """Initialize and start the RealSense camera"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create align object
            self.align = rs.align(rs.stream.color)
            
            # Get camera intrinsics for metadata
            depth_stream = profile.get_stream(rs.stream.depth)
            depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            self.metadata["camera_info"] = {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "model": str(depth_intrinsics.model),
                "coeffs": depth_intrinsics.coeffs,
                "depth_scale": self.depth_scale
            }
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def save_frame(self):
        """Capture and save a single frame"""
        if not self.pipeline:
            return False
        
        try:
            # Get frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return False
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get camera intrinsics
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # Generate point cloud
            points = []
            colors = []
            
            h, w = depth_image.shape
            downsample = 4  # Increase this value for faster processing
            for y in range(0, h, downsample):
                for x in range(0, w, downsample):
                    z = depth_image[y, x] * self.depth_scale
                    if z == 0 or z > 5.0:  # Skip invalid or far points
                        continue
                    
                    # Deproject pixel to 3D point
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], z)
                    points.append(point)
                    
                    # Get color
                    color = color_image[y, x]
                    colors.append(color)
            
            # Save data
            frame_name = f"frame_{self.frame_count:06d}"
            
            # Save point cloud with colors
            if len(points) > 0:
                pointcloud_data = np.column_stack([points, colors])
                np.save(os.path.join(self.pointcloud_dir, f"{frame_name}.npy"), 
                       pointcloud_data.astype(np.float32))
            
            # Save color image
            cv2.imwrite(os.path.join(self.color_dir, f"{frame_name}.jpg"), color_image)
            
            # Save depth image
            np.save(os.path.join(self.depth_dir, f"{frame_name}.npy"), depth_image)
            
            # Update metadata
            self.metadata["frames"].append({
                "frame_id": self.frame_count,
                "timestamp": time.time(),
                "num_points": len(points)
            })
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording and save metadata"""
        if self.pipeline:
            self.pipeline.stop()
        
        # Save metadata
        self.metadata["session_end"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata["total_frames"] = self.frame_count
        
        with open(os.path.join(self.session_dir, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\nRecording saved to: {self.session_dir}")
        print(f"Total frames recorded: {self.frame_count}")

def main():
    print("RealSense Point Cloud Recorder")
    print("-" * 40)
    
    # Check for RealSense devices
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("Error: No RealSense device detected!")
        return
    
    print(f"Found {len(devices)} RealSense device(s)")
    
    # Create recorder
    recorder = RealSenseRecorder()
    
    # Start camera
    if not recorder.start_camera():
        return
    
    # Recording frequency
    RECORDING_FPS = 10 
    frame_interval = 1.0 / RECORDING_FPS
    
    print("\nRecording started. Press Ctrl+C to stop.")
    print(f"Saving frames at {RECORDING_FPS} Hz...\n")
    
    try:
        while True:
            start_time = time.time()
            
            # Save frame
            if recorder.save_frame():
                print(f"Saved frame {recorder.frame_count}", end='\r')
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
    finally:
        recorder.stop_recording()

if __name__ == "__main__":
    main()