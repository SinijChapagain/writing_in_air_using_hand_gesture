import cv2
import torch
import mediapipe as mp
import numpy as np
import joblib
import time
from model import GestureNet


class GestureDemo:
    def __init__(self):
        self.conf_thresh = 0.6
        self.buffer = []
        self.points = []
        self.erase_start_time = None
        self.img_canvas = None

        self.model = self._load_model()
        self.le = joblib.load("label_encoder.pkl")
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def _load_model(self) -> GestureNet:
        model = GestureNet(num_classes=2)
        model.load_state_dict(torch.load("gesture_model.pth", map_location="cpu"))
        model.eval()
        return model

    def _try_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        for prop_id, value in [
            (cv2.CAP_PROP_FRAME_WIDTH, 640),
            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_BUFFERSIZE, 1)
        ]:
            cap.set(prop_id, value)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Failed to open built-in camera (index 0).")

        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[INFO] MacBook camera opened: {frame.shape}")
                return cap
            time.sleep(0.1)

        cap.release()
        raise RuntimeError("Camera opened but no valid frames received.")

    def run(self):
        cap = self._try_camera()
        print("[INFO] Using MacBook built-in camera. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.02)
                    continue

                frame = cv2.resize(frame, (640, 480))
                frame = cv2.flip(frame, 1)
                if self.img_canvas is None or self.img_canvas.shape != frame.shape:
                    self.img_canvas = np.zeros_like(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.mp_hands.process(rgb)
                gesture = "idle"
                erase_center = None

                try:
                    if result.multi_hand_landmarks:
                        lm = result.multi_hand_landmarks[0]
                        vec = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
                        self.buffer.append(vec)
                        if len(self.buffer) > 16:
                            self.buffer.pop(0)

                        if len(self.buffer) == 16:
                            seq = np.array(self.buffer, dtype=np.float32)
                            if seq.shape != (16, 63):
                                continue
                            seq = np.transpose(seq, (1, 0))
                            seq_tensor = torch.tensor(seq).unsqueeze(0)
                            with torch.no_grad():
                                logits = self.model(seq_tensor)
                                pred = logits.argmax(dim=1).item()
                                gesture = self.le.inverse_transform([pred])[0]
                            
                            # Pinch validation for "note"
                            if gesture == "note":
                                thumb_x, thumb_y = lm.landmark[4].x, lm.landmark[4].y
                                index_x, index_y = lm.landmark[8].x, lm.landmark[8].y
                                pinch_dist = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5
                                wrist_x, wrist_y = lm.landmark[0].x, lm.landmark[0].y
                                pinky_x, pinky_y = lm.landmark[17].x, lm.landmark[17].y
                                hand_width = ((wrist_x - pinky_x)**2 + (wrist_y - pinky_y)**2)**0.5
                                if pinch_dist > 0.25 * hand_width:
                                    gesture = "idle"
                            
                            # Open palm validation for "erase"
                            if gesture == "erase":
                                index_x, index_y = lm.landmark[8].x, lm.landmark[8].y
                                pinky_x, pinky_y = lm.landmark[20].x, lm.landmark[20].y
                                finger_spread = ((index_x - pinky_x)**2 + (index_y - pinky_y)**2)**0.5
                                wrist_x, wrist_y = lm.landmark[0].x, lm.landmark[0].y
                                pinky_mcp_x, pinky_mcp_y = lm.landmark[17].x, lm.landmark[17].y
                                hand_width = ((wrist_x - pinky_mcp_x)**2 + (wrist_y - pinky_mcp_y)**2)**0.5
                                if finger_spread < 0.4 * hand_width:
                                    gesture = "idle"
                                erase_center = (
                                    int(lm.landmark[12].x * frame.shape[1]),
                                    int(lm.landmark[12].y * frame.shape[0])
                                )

                    # Drawing logic
                    if gesture == "note":
                        ix = int(lm.landmark[8].x * frame.shape[1])
                        iy = int(lm.landmark[8].y * frame.shape[0])
                        pt = (ix, iy)
                        if not self.points or np.linalg.norm(
                            np.array(self.points[-1]) - np.array(pt)) > 2:
                            self.points.append(pt)
                        for i in range(1, len(self.points)):
                            cv2.line(self.img_canvas, self.points[i-1], self.points[i], (255, 0, 0), 8)

                    elif gesture == "erase" and erase_center:
                        cv2.circle(frame, erase_center, 36, (0, 0, 255), 2)
                        now = time.time()
                        if self.erase_start_time is None:
                            self.erase_start_time = now
                        elif now - self.erase_start_time >= 0.3:
                            cv2.circle(self.img_canvas, erase_center, 36, (0, 0, 0), -1)
                    else:
                        self.erase_start_time = None

                    if gesture != "note":
                        self.points.clear()

                    overlay = cv2.addWeighted(frame, 0.6, self.img_canvas, 0.4, 0)
                    cv2.putText(overlay, f"Gesture: {gesture}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Gesture Notes", overlay)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(f"[ERROR] Frame processing failed: {e}")
                    continue

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    demo = GestureDemo()
    demo.run()