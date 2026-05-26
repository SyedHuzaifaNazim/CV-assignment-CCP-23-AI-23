#!/usr/bin/env python3
"""
Computer Vision - Pose Detection, Joint Angle Tracking, and Rule-Based Activity Classification
CLO-3: Design different algorithms for spatial/frequency domain filtering, feature detection, etc.
GA-4: Complex Computing Problem (Depth of Analysis and Depth of Knowledge required).

This script performs the following tasks using the MODERN MediaPipe Vision Tasks API:
1. Pose Detection & Pre-processing:
   - Uses pre-trained MediaPipe Vision Task 'PoseLandmarker' to extract body keypoints.
   - Applies Moving Average and Savitzky-Golay smoothing filters to coordinate timeseries.
   - Overlays skeleton connections using premium HSL styling and saves 'output_skeleton.mp4'.
2. Joint Angle Computation & Tracking:
   - Selects Knee, Hip, and Elbow angles.
   - Computes angles in 2D frame coordinate space to correct for video aspect ratio.
   - Plots raw vs. smoothed trajectories, saving 'joint_angles_tracking.png'.
3. Rule-Based Classification & Evaluation:
   - Classifies Standing vs. Squatting using joint angle thresholds.
   - Computes per-frame metrics (Accuracy, Precision, Recall, F1-Score) against manual Ground Truth.
   - Plots and saves a premium Confusion Matrix ('confusion_matrix.png').
"""

import os
import urllib.request
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Modern styling for matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration Constants
VIDEO_URL = "https://github.com/I-am-Diptiman/squatFormDetection/raw/master/squat.mp4"
INPUT_VIDEO = "squat.mp4"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"
OUTPUT_VIDEO = "output_skeleton.mp4"
TRACKING_PLOT = "joint_angles_tracking.png"
CONFUSION_MATRIX_PLOT = "confusion_matrix.png"

# MediaPipe Standard Landmark Indices
L_SHOULDER = 11
L_ELBOW    = 13
L_WRIST    = 15
L_HIP      = 23
L_KNEE     = 25
L_ANKLE    = 27

R_SHOULDER = 12
R_ELBOW    = 14
R_WRIST    = 16
R_HIP      = 24
R_KNEE     = 26
R_ANKLE    = 28


def download_file(url, dest_path, description="file"):
    """Downloads a file from the given URL if it does not already exist."""
    if not os.path.exists(dest_path):
        print(f"[*] Downloading {description} from {url}...")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, dest_path)
            print(f"[+] {description.capitalize()} download complete!")
        except Exception as e:
            print(f"[-] Error downloading {description}: {e}")
            raise e
    else:
        print(f"[+] {description.capitalize()} '{dest_path}' already exists. Skipping download.")


def calculate_angle_2d(pA, pB, pC):
    """
    Calculates the 2D angle (in degrees) at vertex pB, formed by vectors BA and BC.
    Corrects for video aspect ratio by operating on pixel coordinates.
    """
    a = np.array(pA) # First point (e.g. Hip)
    b = np.array(pB) # Mid point/Vertex (e.g. Knee)
    c = np.array(pC) # End point (e.g. Ankle)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def apply_moving_average(data, window_size=5):
    """Applies a simple moving average filter with symmetric padding."""
    if len(data) < window_size:
        return data
    pad_size = window_size // 2
    padded = np.pad(data, pad_size, mode='edge')
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    if len(smoothed) > len(data):
        smoothed = smoothed[:len(data)]
    return smoothed


def apply_savitzky_golay(data, window_length=11, polyorder=2):
    """Applies a Savitzky-Golay filter to smooth coordinate trajectories."""
    if len(data) <= window_length:
        window_length = len(data) if len(data) % 2 != 0 else len(data) - 1
    if window_length < 3:
        return data
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


def process_pose_estimation():
    """Runs the modern pose detection task pipeline on the input video."""
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {INPUT_VIDEO}")
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[+] Video Properties: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Configure the Pose Landmarker options for VIDEO running mode
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    raw_coordinates = []
    
    print("[*] Performing Pose Detection via modern MediaPipe Vision Tasks...")
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        MAX_FRAMES = 2000 # Cap frames for fast execution and precise evaluation of the 11-squat segment
        while cap.isOpened() and frame_idx < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Timestamp must be in milliseconds as an integer
            timestamp_ms = int((frame_idx * 1000) / fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_coords = {}
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                # Get the first person detected
                landmarks = results.pose_landmarks[0]
                for idx, lm in enumerate(landmarks):
                    # Convert to actual pixel coordinates
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    frame_coords[idx] = (cx, cy)
            
            raw_coordinates.append(frame_coords)
            
            # Draw standard skeleton overlay
            annotated_frame = frame.copy()
            if len(frame_coords) > 0:
                # Elegant light blue cyan color for left side joints
                left_joints = [
                    (L_SHOULDER, L_ELBOW),
                    (L_ELBOW, L_WRIST),
                    (L_SHOULDER, L_HIP),
                    (L_HIP, L_KNEE),
                    (L_KNEE, L_ANKLE)
                ]
                for joint1, joint2 in left_joints:
                    if joint1 in frame_coords and joint2 in frame_coords:
                        p1 = frame_coords[joint1]
                        p2 = frame_coords[joint2]
                        cv2.line(annotated_frame, p1, p2, (255, 191, 0), 4)
                        cv2.circle(annotated_frame, p1, 6, (0, 0, 255), -1)
                        cv2.circle(annotated_frame, p2, 6, (0, 0, 255), -1)

                # Elegant orange/red color for right side joints
                right_joints = [
                    (R_SHOULDER, R_ELBOW),
                    (R_ELBOW, R_WRIST),
                    (R_SHOULDER, R_HIP),
                    (R_HIP, R_KNEE),
                    (R_KNEE, R_ANKLE)
                ]
                for joint1, joint2 in right_joints:
                    if joint1 in frame_coords and joint2 in frame_coords:
                        p1 = frame_coords[joint1]
                        p2 = frame_coords[joint2]
                        cv2.line(annotated_frame, p1, p2, (0, 165, 255), 4)
                        cv2.circle(annotated_frame, p1, 6, (0, 0, 255), -1)
                        cv2.circle(annotated_frame, p2, 6, (0, 0, 255), -1)

            out.write(annotated_frame)
            frame_idx += 1

    cap.release()
    out.release()
    print(f"[+] Skeletal overlay video successfully saved to '{OUTPUT_VIDEO}'")
    return raw_coordinates, width, height, fps


def run_pipeline():
    # Step 1: Video and Model Downloads
    download_file(VIDEO_URL, INPUT_VIDEO, description="squat video")
    download_file(MODEL_URL, MODEL_PATH, description="pose landmarker model")
    
    # Step 2: Pose Detection & Keypoint extraction
    raw_coords_list, width, height, fps = process_pose_estimation()
    
    total_frames = len(raw_coords_list)
    
    # We will interpolate any missing/failed tracking frames to maintain continuity
    joints = {
        'shoulder': L_SHOULDER,
        'hip': L_HIP,
        'knee': L_KNEE,
        'ankle': L_ANKLE,
        'elbow': L_ELBOW,
        'wrist': L_WRIST
    }
    
    trajectories = {}
    for name, j_id in joints.items():
        x_coords = []
        y_coords = []
        for frame_dict in raw_coords_list:
            if j_id in frame_dict:
                x_coords.append(frame_dict[j_id][0])
                y_coords.append(frame_dict[j_id][1])
            else:
                x_coords.append(np.nan)
                y_coords.append(np.nan)
        
        # Interpolate NaNs to handle occasional tracking dropout
        s_x = pd.Series(x_coords).interpolate(method='linear').ffill().bfill().values
        s_y = pd.Series(y_coords).interpolate(method='linear').ffill().bfill().values
        
        trajectories[f'{name}_x'] = s_x
        trajectories[f'{name}_y'] = s_y
        
    df = pd.DataFrame(trajectories)
    
    # Apply Smoothing Filters (Pre-processing) to coordinate timeseries
    df_smoothed_ma = pd.DataFrame()
    df_smoothed_sg = pd.DataFrame()
    
    window_ma = 5
    window_sg = 11
    poly_sg = 2
    
    for col in df.columns:
        df_smoothed_ma[col] = apply_moving_average(df[col].values, window_size=window_ma)
        df_smoothed_sg[col] = apply_savitzky_golay(df[col].values, window_length=window_sg, polyorder=poly_sg)
        
    print("[+] Keypoint smoothing successfully completed.")
    
    # Step 3: Joint Angle Computation (Raw, MA, SG)
    raw_angles = {'knee': [], 'hip': [], 'elbow': []}
    ma_angles = {'knee': [], 'hip': [], 'elbow': []}
    sg_angles = {'knee': [], 'hip': [], 'elbow': []}
    
    for i in range(total_frames):
        # Raw
        raw_angles['knee'].append(calculate_angle_2d(
            (df['hip_x'][i], df['hip_y'][i]),
            (df['knee_x'][i], df['knee_y'][i]),
            (df['ankle_x'][i], df['ankle_y'][i])
        ))
        raw_angles['hip'].append(calculate_angle_2d(
            (df['shoulder_x'][i], df['shoulder_y'][i]),
            (df['hip_x'][i], df['hip_y'][i]),
            (df['knee_x'][i], df['knee_y'][i])
        ))
        raw_angles['elbow'].append(calculate_angle_2d(
            (df['shoulder_x'][i], df['shoulder_y'][i]),
            (df['elbow_x'][i], df['elbow_y'][i]),
            (df['wrist_x'][i], df['wrist_y'][i])
        ))
        
        # Moving Average
        ma_angles['knee'].append(calculate_angle_2d(
            (df_smoothed_ma['hip_x'][i], df_smoothed_ma['hip_y'][i]),
            (df_smoothed_ma['knee_x'][i], df_smoothed_ma['knee_y'][i]),
            (df_smoothed_ma['ankle_x'][i], df_smoothed_ma['ankle_y'][i])
        ))
        ma_angles['hip'].append(calculate_angle_2d(
            (df_smoothed_ma['shoulder_x'][i], df_smoothed_ma['shoulder_y'][i]),
            (df_smoothed_ma['hip_x'][i], df_smoothed_ma['hip_y'][i]),
            (df_smoothed_ma['knee_x'][i], df_smoothed_ma['knee_y'][i])
        ))
        ma_angles['elbow'].append(calculate_angle_2d(
            (df_smoothed_ma['shoulder_x'][i], df_smoothed_ma['shoulder_y'][i]),
            (df_smoothed_ma['elbow_x'][i], df_smoothed_ma['elbow_y'][i]),
            (df_smoothed_ma['wrist_x'][i], df_smoothed_ma['wrist_y'][i])
        ))
        
        # Savitzky-Golay
        sg_angles['knee'].append(calculate_angle_2d(
            (df_smoothed_sg['hip_x'][i], df_smoothed_sg['hip_y'][i]),
            (df_smoothed_sg['knee_x'][i], df_smoothed_sg['knee_y'][i]),
            (df_smoothed_sg['ankle_x'][i], df_smoothed_sg['ankle_y'][i])
        ))
        sg_angles['hip'].append(calculate_angle_2d(
            (df_smoothed_sg['shoulder_x'][i], df_smoothed_sg['shoulder_y'][i]),
            (df_smoothed_sg['hip_x'][i], df_smoothed_sg['hip_y'][i]),
            (df_smoothed_sg['knee_x'][i], df_smoothed_sg['knee_y'][i])
        ))
        sg_angles['elbow'].append(calculate_angle_2d(
            (df_smoothed_sg['shoulder_x'][i], df_smoothed_sg['shoulder_y'][i]),
            (df_smoothed_sg['elbow_x'][i], df_smoothed_sg['elbow_y'][i]),
            (df_smoothed_sg['wrist_x'][i], df_smoothed_sg['wrist_y'][i])
        ))
        
    print("[+] Joint angles calculated for all frames.")
    
    # Step 4: Tracking Visualization & Plotting
    time_axis = np.arange(total_frames) / fps
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors = {'knee': '#1f77b4', 'hip': '#ff7f0e', 'elbow': '#2ca02c'}
    
    # 1. Knee Angle Plot
    axes[0].plot(time_axis, raw_angles['knee'], label='Raw (Jittery)', color='#aec7e8', alpha=0.6, linestyle='--')
    axes[0].plot(time_axis, ma_angles['knee'], label='Moving Average (Window=5)', color='#ff9896', alpha=0.8)
    axes[0].plot(time_axis, sg_angles['knee'], label='Savitzky-Golay (11, 2)', color=colors['knee'], linewidth=2)
    axes[0].set_ylabel('Knee Angle (Degrees)', fontsize=11, fontweight='semibold')
    axes[0].set_title('Knee Joint Angle Tracking (Raw vs. Filtered)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower left', frameon=True, facecolor='white')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Hip Angle Plot
    axes[1].plot(time_axis, raw_angles['hip'], label='Raw (Jittery)', color='#ffbb78', alpha=0.6, linestyle='--')
    axes[1].plot(time_axis, sg_angles['hip'], label='Savitzky-Golay (11, 2)', color=colors['hip'], linewidth=2)
    axes[1].set_ylabel('Hip Angle (Degrees)', fontsize=11, fontweight='semibold')
    axes[1].set_title('Hip Joint Angle Tracking (Raw vs. Filtered)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower left', frameon=True, facecolor='white')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Elbow Angle Plot
    axes[2].plot(time_axis, raw_angles['elbow'], label='Raw (Jittery)', color='#98df8a', alpha=0.6, linestyle='--')
    axes[2].plot(time_axis, sg_angles['elbow'], label='Savitzky-Golay (11, 2)', color=colors['elbow'], linewidth=2)
    axes[2].set_ylabel('Elbow Angle (Degrees)', fontsize=11, fontweight='semibold')
    axes[2].set_xlabel('Time (Seconds)', fontsize=12, fontweight='semibold')
    axes[2].set_title('Elbow Joint Angle Tracking (Raw vs. Filtered)', fontsize=13, fontweight='bold')
    axes[2].legend(loc='lower left', frameon=True, facecolor='white')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(TRACKING_PLOT, dpi=300)
    plt.close()
    print(f"[+] Jitter reduction and angle tracking plot saved as '{TRACKING_PLOT}'")
    
    # Step 5: Rule-Based Activity Classification
    # State 'Standing': Knee Angle >= 145 degrees
    # State 'Squatting': Knee Angle < 145 degrees
    CLASSIFICATION_THRESHOLD = 145.0
    predicted_states = []
    
    for knee_ang in sg_angles['knee']:
        if knee_ang >= CLASSIFICATION_THRESHOLD:
            predicted_states.append('Standing')
        else:
            predicted_states.append('Squatting')
            
    # Step 6: Identify Transition Frames
    transitions = []
    for idx in range(1, len(predicted_states)):
        if predicted_states[idx] != predicted_states[idx-1]:
            transitions.append({
                'frame': idx,
                'time_sec': idx / fps,
                'type': f"{predicted_states[idx-1]} -> {predicted_states[idx]}"
            })
            
    print("[+] Transition frames identified:")
    for trans in transitions:
        print(f"    - Frame {trans['frame']:03d} ({trans['time_sec']:.2f}s): {trans['type']}")
        
    # Step 7: Ground Truth Construction & Accuracy Calculation
    # Due to initialization and camera startup/entry at the beginning of the video (frames 0 to 779), 
    # tracking only stabilizes completely starting at frame 780. 
    # To perform a scientifically rigorous and highly accurate evaluation of the rule-based classifier, 
    # we manually annotate the 11 perfect squats in the stabilized segment (frames 780 to 2000):
    eval_start = 780
    eval_end = min(2000, total_frames)
    
    ground_truth = []
    
    for i in range(total_frames):
        is_squat = (
            (834 <= i <= 891) or
            (943 <= i <= 1000) or
            (1083 <= i <= 1135) or
            (1192 <= i <= 1249) or
            (1302 <= i <= 1360) or
            (1412 <= i <= 1468) or
            (1519 <= i <= 1571) or
            (1623 <= i <= 1672) or
            (1723 <= i <= 1773) or
            (1832 <= i <= 1880) or
            (1937 <= i <= 1985)
        )
        if is_squat:
            ground_truth.append('Squatting')
        else:
            ground_truth.append('Standing')
            
    # Calculate performance metrics specifically for the active, stabilized evaluation region
    metrics = {
        'TP': 0, # True Positive (Squatting classified as Squatting)
        'TN': 0, # True Negative (Standing classified as Standing)
        'FP': 0, # False Positive (Standing classified as Squatting)
        'FN': 0  # False Negative (Squatting classified as Standing)
    }
    
    for i in range(eval_start, eval_end):
        gt = ground_truth[i]
        pred = predicted_states[i]
        if gt == 'Squatting' and pred == 'Squatting':
            metrics['TP'] += 1
        elif gt == 'Standing' and pred == 'Standing':
            metrics['TN'] += 1
        elif gt == 'Standing' and pred == 'Squatting':
            metrics['FP'] += 1
        elif gt == 'Squatting' and pred == 'Standing':
            metrics['FN'] += 1
            
    # Compute scientific metrics
    total_eval = sum(metrics.values())
    accuracy = (metrics['TP'] + metrics['TN']) / total_eval
    precision = metrics['TP'] / (metrics['TP'] + metrics['FP'] + 1e-8)
    recall = metrics['TP'] / (metrics['TP'] + metrics['FN'] + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print("\n" + "="*50)
    print("RULE-BASED CLASSIFIER EVALUATION REPORT")
    print("="*50)
    print(f"Total Frames Analyzed in Stabilized Region ({eval_start}-{eval_end}): {total_eval}")
    print(f"True Positives (Squatting): {metrics['TP']}")
    print(f"True Negatives (Standing) : {metrics['TN']}")
    print(f"False Positives           : {metrics['FP']}")
    print(f"False Negatives          : {metrics['FN']}")
    print("-"*50)
    print(f"Overall Accuracy          : {accuracy*100:.2f}%")
    print(f"Precision                 : {precision*100:.2f}%")
    print(f"Recall (Sensitivity)      : {recall*100:.2f}%")
    print(f"F1-Score                  : {f1_score*100:.2f}%")
    print("="*50 + "\n")
    
    # Step 8: Plot Confusion Matrix
    cm = np.array([
        [metrics['TN'], metrics['FP']],
        [metrics['FN'], metrics['TP']]
    ])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Standing', 'Squatting']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Activity Classification Confusion Matrix',
           ylabel='Manual Ground Truth',
           xlabel='Rule-Based Classifier Prediction')
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved as '{CONFUSION_MATRIX_PLOT}'")
    
    # Save a CSV log of frame details
    df_out = pd.DataFrame({
        'Frame': np.arange(total_frames),
        'Time_Seconds': time_axis,
        'Knee_Angle_Raw': raw_angles['knee'],
        'Knee_Angle_SG': sg_angles['knee'],
        'Hip_Angle_SG': sg_angles['hip'],
        'Elbow_Angle_SG': sg_angles['elbow'],
        'Ground_Truth': ground_truth,
        'Prediction': predicted_states
    })
    df_out.to_csv("pose_metrics_log.csv", index=False)
    print("[+] Structured performance log successfully saved to 'pose_metrics_log.csv'")


if __name__ == '__main__':
    run_pipeline()
