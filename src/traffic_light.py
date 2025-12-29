import cv2
import numpy as np
from collections import deque, Counter
from src.config import *

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DEBUG_VISUALIZATION = False
VOTING_WINDOW = 4   # number of frames for temporal voting

state_buffer = deque(maxlen=VOTING_WINDOW)


# --------------------------------------------------
# Main detection function
# --------------------------------------------------
def detect_and_classify_traffic_light(frame):
    h, w, _ = frame.shape

    # ---------------- ROI DEFINITION ----------------
    y1, y2 = 0, int(0.3 * h)
    x1, x2 = int(0.2 * w), int(0.8 * w)

    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ---------------- Color masks (ROI ONLY) ----------------
    red_mask1 = cv2.inRange(hsv_roi, RED_LOWER_1, RED_UPPER_1)
    red_mask2 = cv2.inRange(hsv_roi, RED_LOWER_2, RED_UPPER_2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv_roi, YELLOW_LOWER, YELLOW_UPPER)
    #cv2.imshow("Yellow Mask", yellow_mask)
    #cv2.waitKey(0)
    green_mask = cv2.inRange(hsv_roi, GREEN_LOWER, GREEN_UPPER)
    #show green mask
    #cv2.imshow("Green Mask", green_mask)
    #cv2.waitKey(0)

    # ---------------- Process each color ----------------
    red_detected = _process_color(frame, hsv_roi, red_mask, "RED", x1, y1)
    yellow_detected = _process_color(frame, hsv_roi, yellow_mask, "YELLOW", x1, y1)
    green_detected = _process_color(frame, hsv_roi, green_mask, "GREEN", x1, y1)

    # ---------------- Frame-level decision ----------------
    if red_detected:
        state = "RED"
    elif yellow_detected:
        state = "YELLOW"
    elif green_detected:
        state = "GREEN"
    else:
        state = "UNKNOWN"

    # ---------------- Temporal voting ----------------
    state_buffer.append(state)

    if len(state_buffer) > 0:
        voted_state = Counter(state_buffer).most_common(1)[0][0]
    else:
        voted_state = "UNKNOWN"

    # ---------------- Visualization of final state ----------------
    cv2.putText(
        frame,
        f"TL STATE: {voted_state}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    return frame, voted_state


# --------------------------------------------------
# Color-specific processing
# --------------------------------------------------
def _process_color(frame, hsv_roi, mask, label, x_offset, y_offset):
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(f"Contour area for {label}: {area}")
        if area < 30:
            continue

        contour_mask = np.zeros(clean_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        mean_h, mean_s, mean_v, _ = cv2.mean(hsv_roi, mask=contour_mask)

        perimeter = cv2.arcLength(cnt, True)
        circularity = (
            4 * np.pi * area / (perimeter ** 2 + 1e-6)
            if perimeter > 0 else 0
        )

        # ---------------- Validation rules ----------------
        pass_area = 30 < area < 900
        pass_brightness = mean_v > 120
        pass_circularity = circularity > 0.75

        passed = pass_area and pass_brightness and pass_circularity

        print(
            f"{label} | A:{area:.1f} "
            f"V:{mean_v:.1f} "
            f"C:{circularity:.2f} -> "
            f"{'PASS' if passed else 'FAIL'}"
        )

        # ---------------- Visualization ----------------
        if DEBUG_VISUALIZATION or passed:
            box_color = (
                (0, 0, 255) if label == "RED" else
                (0, 255, 255) if label == "YELLOW" else
                (0, 255, 0)
            )

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                frame,
                (x + x_offset, y + y_offset),
                (x + w + x_offset, y + h + y_offset),
                box_color,
                2
            )

            if passed:
                cv2.putText(
                    frame,
                    label,
                    (x + x_offset, y + y_offset - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

            elif DEBUG_VISUALIZATION:
                cv2.putText(
                    frame,
                    f"{label}-FAIL",
                    (x + x_offset, y + y_offset - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1
                )

        if passed:
            detected = True

    return detected


# --------------------------------------------------
# Standalone image test
# --------------------------------------------------
if __name__ == "__main__":
    import glob

    image_paths = glob.glob(
        "data/frames/traffic_light_frames/false_lights/Image 9.png"
    )

    print(f"Found {len(image_paths)} test images")

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        frame, state = detect_and_classify_traffic_light(frame)

        print(f"{path} → {state}")
        cv2.imshow("Traffic Light Debug", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
