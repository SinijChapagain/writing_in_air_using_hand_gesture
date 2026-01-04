import cv2
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def record_clip(base_filename: str, duration_sec: int = 12) -> str | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error(
            "Camera access failed. Check System Settings → Privacy & Security → Camera."
        )
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.mp4"
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))

    logger.info(f"Recording '{filename}' for {duration_sec} seconds. Press 'q' to stop.")
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed. Exiting.")
                break
            out.write(frame)
            cv2.imshow("Recording — Press 'q' to quit", frame)

            elapsed = time.time() - start_time
            if cv2.waitKey(1) & 0xFF == ord("q") or elapsed >= duration_sec:
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    logger.info(f"Recording saved: {filename}")
    return filename


if __name__ == "__main__":
    record_clip("pinch_note", duration_sec=12)
    #record_clip("clip_erase", duration_sec=12)
    #record_clip("clip_idle", duration_sec=12)
