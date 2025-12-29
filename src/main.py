import cv2
from src.lane_detection import detect_lanes
from src.barrier_detection import detect_barriers
from src.traffic_light import detect_and_classify_traffic_light
from src.obstacle_detection import detect_obstacles

VIDEO_PATH = "data/videos/input.mp4"

SAVE_OUTPUT = False
OUTPUT_VIDEO_PATH = "outputs/Final.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    writer = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if SAVE_OUTPUT and writer is None:
            h, w, _ = frame.shape
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                OUTPUT_VIDEO_PATH,
                fourcc,
                fps,
                (w, h)
            )

        # ---- TASKS ----
        #frame = detect_barriers(frame)
        frame = detect_lanes(frame)
        frame, _ = detect_and_classify_traffic_light(frame)
        frame, _ = detect_obstacles(frame)
        if SAVE_OUTPUT:
            writer.write(frame)

        cv2.imshow("Autonomous Vision Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if SAVE_OUTPUT and writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
