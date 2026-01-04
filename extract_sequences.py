import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from typing import Tuple


hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)


def extract_landmarks_from_video(video_path: str) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarks = []
    timestamps = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        t = frame_idx / fps

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            vec = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            landmarks.append(vec)
        else:
            landmarks.append(np.full(63, np.nan))
        timestamps.append(t)
        frame_idx += 1

    cap.release()
    return np.array(landmarks), np.array(timestamps)


def extract_sequences(
    video_path: str,
    start_sec: float,
    end_sec: float,
    window_size: int = 16,
    step: int = 4
) -> np.ndarray:
    landmarks, times = extract_landmarks_from_video(video_path)
    mask = (times >= start_sec) & (times <= end_sec)
    segment = landmarks[mask]

    #Remove frames with missing landmarks
    valid_mask = ~np.isnan(segment).any(axis=1)
    segment = segment[valid_mask]

    if len(segment) < window_size:
        return np.empty((0, window_size, 63))

    sequences = []
    for i in range(0, len(segment) - window_size + 1, step):
        seq = segment[i:i + window_size]
        # Only keep sequences with complete landmark data
        if not np.isnan(seq).any():
            sequences.append(seq)
    return np.array(sequences)


if __name__ == "__main__":
    annotations = pd.read_csv("annotations.csv")
    X_seq, y_seq = [], []

    for _, row in annotations.iterrows():
        fname = row["filename"].strip()
        if not os.path.isfile(fname):
            print(f"[WARN] File not found: {fname}")
            continue
        seqs = extract_sequences(
            fname,
            float(row["start_sec"]),
            float(row["end_sec"])
        )
        if seqs.size > 0:
            X_seq.append(seqs)
            y_seq.extend([row["gesture"].strip()] * len(seqs))
        else:
            print(f"[WARN] No valid sequences extracted from {fname}")

    if not X_seq:
        raise ValueError("No valid sequences extracted. Check video paths and annotations.")

    X = np.vstack(X_seq).astype(np.float32)
    y = np.array(y_seq)

    # Per-sequence normalization (removes spatial information)
    # if X.size > 0:
    #     X = (X - X.mean(axis=(1, 2), keepdims=True)) / (
    #         X.std(axis=(1, 2), keepdims=True) + 1e-8
    #     )

    if X.size > 0:
        feature_mean = X.mean(axis=(0, 1), keepdims=True)
        feature_std = X.std(axis=(0, 1), keepdims=True)
        X = (X - feature_mean) / (feature_std + 1e-8)

    print(f"Extracted {X.shape[0]} sequences of shape {X.shape[1:]}")
    print(f"Classes: {np.unique(y)}")

    # Diagnostic: class distribution and feature stats
    for cls in np.unique(y):
        mask = (y == cls)
        count = mask.sum()
        mean_x = X[mask, :, 8].mean()  # index tip x-coordinate
        mean_y = X[mask, :, 9].mean()  # index tip y-coordinate
        idx_x = X[mask, :, 8]
        print(f"[DIAG] '{cls}': {count} samples | index.x range: ({idx_x.min():.3f}, {idx_x.max():.3f}) | mean: {mean_x:.3f}")

    np.save("X_seq.npy", X)
    np.save("y_seq.npy", y)
