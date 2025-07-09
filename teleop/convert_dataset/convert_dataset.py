import os
import json
import shutil
import numpy as np
from pathlib import Path

def convert_episode(episode_dir, output_base_dir, task_name):
    data_json_path = os.path.join(episode_dir, 'data.json')
    if not os.path.exists(data_json_path):
        print(f"Warning: {data_json_path} not found, skipping...")
        return []
    
    with open(data_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Get episode ID
    episode_name = os.path.basename(episode_dir)
    episode_id = episode_name.split('_')[-1]  # e.g.: episode_0001 -> 0001
    
    # Create output directory structure
    rgb_dir = os.path.join(output_base_dir, f"{episode_id}_rgb", episode_id)
    os.makedirs(rgb_dir, exist_ok=True)
    
    converted_items = []
    
    # Get task description
    if task_name:
        task_description = task_name
    else:
        task_description = original_data.get('text', {}).get('goal', 'Unknown task')
    
    # Store previous frame's arm positions for delta calculation
    prev_left_arm = None
    prev_right_arm = None
    
    # Process each data frame
    data_items = original_data.get('data', [])
    for i, item in enumerate(data_items):
        step_id = str(item['idx']).zfill(4)  # e.g.: 0 -> 0000, 1 -> 0001
        
        # Process images
        if 'colors' in item and item['colors']:
            # Use the first camera's image
            color_key = list(item['colors'].keys())[0]  # e.g.: 'color_0'
            original_image_path = os.path.join(episode_dir, item['colors'][color_key])
            
            # New image path
            new_image_name = f"{step_id}.png"
            new_image_path = os.path.join(rgb_dir, new_image_name)
            
            # Copy image file
            if os.path.exists(original_image_path):
                shutil.copy2(original_image_path, new_image_path)
                
                # Build relative path
                relative_image_path = os.path.join(f"{episode_id}_rgb", episode_id, new_image_name)
                
                # Extract action data with delta calculation
                raw_action, current_left_arm, current_right_arm = extract_action_data(
                    item, prev_left_arm, prev_right_arm, is_first_frame=(i == 0)
                )
                
                # Update previous positions for next frame
                prev_left_arm = current_left_arm
                prev_right_arm = current_right_arm
                
                # Create converted data entry
                converted_item = {
                    "image": relative_image_path,
                    "task": task_description,
                    "raw_action": json.dumps(raw_action)  # Convert to JSON string format
                }
                
                converted_items.append(converted_item)
            else:
                print(f"Warning: Image file {original_image_path} not found")
    
    return converted_items

def extract_action_data(item, prev_left_arm=None, prev_right_arm=None, is_first_frame=False):
    actions = item.get('actions', {})
    
    # Get current action data
    left_arm = actions.get('left_arm', {}).get('qpos', [])
    left_hand = actions.get('left_hand', {}).get('qpos', [])
    right_arm = actions.get('right_arm', {}).get('qpos', [])
    right_hand = actions.get('right_hand', {}).get('qpos', [])
    
    # Calculate delta positions for arms
    if is_first_frame:
        left_arm_delta = [0.0] * len(left_arm)
        right_arm_delta = [0.0] * len(right_arm)
    else:
        # Calculate increment
        if prev_left_arm is not None and len(left_arm) == len(prev_left_arm):
            left_arm_delta = [current - prev for current, prev in zip(left_arm, prev_left_arm)]
        else:
            print(f"Warning: Previous left arm position unavailable or length mismatch")
            left_arm_delta = [0.0] * len(left_arm)
        
        if prev_right_arm is not None and len(right_arm) == len(prev_right_arm):
            right_arm_delta = [current - prev for current, prev in zip(right_arm, prev_right_arm)]
        else:
            print(f"Warning: Previous right arm position unavailable or length mismatch")
            right_arm_delta = [0.0] * len(right_arm)
    
    # Convert hand states to binary values
    left_hand_state = convert_hand_state(left_hand, 'left')
    right_hand_state = convert_hand_state(right_hand, 'right')
    
    # Combine actions in order: left arm + right arm + left hand + right hand
    combined_action = left_arm_delta + right_arm_delta + [left_hand_state] + [right_hand_state]
    
    # Return action and current positions for next frame's delta calculation
    return combined_action, left_arm, right_arm

def convert_hand_state(hand_qpos, hand_type):
    """Convert hand qpos to binary state (0=open, 1=closed)"""
    # Define expected hand states
    left_hand_closed = np.array([0, 1.05, 1.75, -1.57, -1.75, -1.57, -1.75])
    right_hand_closed = np.array([0, -1.05, -1.75, 1.57, 1.75, 1.57, 1.75])
    
    # Check if all values are zero (open state)
    if all(abs(val) < 1e-6 for val in hand_qpos):
        return 0  # Open
    
    # Check for specific closed hand states
    if hand_type == 'left':
        if len(hand_qpos) == 7 and all(abs(a - b) < 1e-3 for a, b in zip(hand_qpos, left_hand_closed)):
            return 1  # Closed
    elif hand_type == 'right':
        if len(hand_qpos) == 7 and all(abs(a - b) < 1e-3 for a, b in zip(hand_qpos, right_hand_closed)):
            return 1  # Closed
    
    # If doesn't match expected states, raise error
    raise ValueError(f"Unexpected {hand_type} hand state: {hand_qpos}. "
                    f"Expected all zeros (open), left_hand_closed {left_hand_closed}, "
                    f"or right_hand_closed {right_hand_closed}")

def convert_dataset(input_dir, output_dir, task_name=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Require task name
    if task_name is None:
        raise ValueError("task_name is required. Please specify --task_name argument.")
    
    all_converted_data = []
    
    # Find all episode directories
    episode_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    print(f"Found {len(episode_dirs)} episodes to convert...")
    
    for episode_dir in episode_dirs:
        print(f"Converting {episode_dir.name}...")
        
        converted_items = convert_episode(
            str(episode_dir), 
            str(output_path), 
            task_name
        )
        
        all_converted_data.extend(converted_items)
        print(f"  -> Converted {len(converted_items)} items")
    
    # Save converted data
    output_json_path = output_path / "converted_dataset.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversion completed!")
    print(f"Total items: {len(all_converted_data)}")
    print(f"Output JSON: {output_json_path}")
    print(f"Output directory structure created in: {output_path}")

def main():
    import config_converter
    convert_dataset(config_converter.INPUT_DIR, config_converter.OUTPUT_DIR, config_converter.TASK_NAME)

if __name__ == "__main__":
    main()