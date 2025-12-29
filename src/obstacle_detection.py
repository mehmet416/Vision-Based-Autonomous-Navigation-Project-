import cv2
import numpy as np
from collections import deque

DEBUG_VISUALIZATION = False

# --------------------------------------------------
# Temporal voting buffer
# --------------------------------------------------
VOTING_WINDOW = 1
VOTING_THRESHOLD = 1
obstacle_history = deque(maxlen=VOTING_WINDOW)


# --------------------------------------------------
# ROI mask
# --------------------------------------------------
def get_roi_mask(frame):
    h, w = frame.shape[:2]

    roi_pts = np.array([[
        (int(0.25 * w), int(0.7 * h)),
        (int(0.75 * w), int(0.7 * h)),
        (int(0.75 * w), int(0.25 * h)),
        (int(0.25 * w), int(0.25 * h))
    ]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, roi_pts, 255)

    return mask, roi_pts


# --------------------------------------------------
# Main obstacle detector
# --------------------------------------------------
def detect_obstacles(frame):
    h, w = frame.shape[:2]
    output = frame.copy()

    # ---------------- ROI ----------------
    roi_mask, roi_pts = get_roi_mask(frame)

    # ---------------- Preprocessing ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # ---------------- Vertical Edge Bins ----------------
    num_bins = 24
    bin_width = w // num_bins
    density = np.zeros(num_bins)

    for i in range(num_bins):
        x1 = i * bin_width
        x2 = (i + 1) * bin_width
        density[i] = cv2.countNonZero(edges[:, x1:x2])

    mean_d = np.mean(density)
    std_d = np.std(density)
    active_bins = density > (mean_d + 1.2 * std_d)

    # ---------------- Group Active Bins ----------------
    candidates = []
    visited = np.zeros(num_bins, dtype=bool)

    for i in range(num_bins):
        if not active_bins[i] or visited[i]:
            continue

        left, right = i, i
        while left > 0 and active_bins[left - 1]:
            left -= 1
        while right < num_bins - 1 and active_bins[right + 1]:
            right += 1

        visited[left:right + 1] = True

        x1 = left * bin_width
        x2 = (right + 1) * bin_width
        bw = x2 - x1

        band = edges[:, x1:x2]

        # ---------------- Vertical Localization (FIX) ----------------
        row_density = np.sum(band > 0, axis=1).astype(np.float32)

        # Smooth to suppress noise
        row_density = cv2.GaussianBlur(
            row_density.reshape(-1, 1),
            (1, 21),
            0
        ).flatten()

        rd_mean = np.mean(row_density)
        rd_std = np.std(row_density)
        active_rows = row_density > (rd_mean + 1.0 * rd_std)

        if np.count_nonzero(active_rows) < 30:
            continue

        ys = np.where(active_rows)[0]
        y1 = int(ys[0])
        y2 = int(ys[-1])
        bh = y2 - y1

        # ---------------- Validation ----------------
        pass_width = bw > 0.08 * w
        pass_height = bh > 0.12 * h
        pass_bottom = y2 > 0.6 * h
        pass_aspect = bh / (bw + 1e-6) > 0.6

        density_score = np.sum(active_rows) / (bh + 1e-6)
        pass_density = density_score > 0.08

        passed = (
            pass_width and
            pass_height and
            pass_bottom and
            pass_aspect and
            pass_density
        )

        if passed:
            candidates.append((x1, y1, bw, bh))

    # ---------------- Temporal Voting ----------------
    obstacle_history.append(candidates)

    confirmed = []
    for bx, by, bw, bh in candidates:
        votes = 0
        for past in obstacle_history:
            for px, py, pw, ph in past:
                if (
                    min(bx + bw, px + pw) > max(bx, px) and
                    min(by + bh, py + ph) > max(by, py)
                ):
                    votes += 1
                    break

        if votes >= VOTING_THRESHOLD:
            confirmed.append((bx, by, bw, bh))

    # ---------------- Visualization ----------------
    for (x, y, bw, bh) in confirmed:
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (255, 0, 0), 3)
        cv2.putText(output, "OBSTACLE",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0), 2)

    if DEBUG_VISUALIZATION:
        overlay = output.copy()
        cv2.polylines(overlay, roi_pts, True, (0, 0, 250), 6)
        cv2.addWeighted(overlay, 0.8, output, 0.6, 0, output)
        cv2.imshow("Edges (ROI)", edges)
        #show vertical bins as bar graph and color the active bins in blue and then show the active bins in the image
        bin_vis = np.zeros((200, w, 3), dtype=np.uint8)
        for i in range(num_bins): 
            bin_height = int((density[i] / (np.max(density) + 1e-6)) * 200)
            color = (255, 0, 0) if active_bins[i] else (100, 100, 100)
            cv2.rectangle(bin_vis,
                          (i * bin_width, 200 - bin_height),
                          ((i + 1) * bin_width - 1, 200),
                          color,
                          -1)
        cv2.imshow("Vertical Edge Density Bins", bin_vis)       

        


    return output, confirmed


# --------------------------------------------------
# Test
# --------------------------------------------------
if __name__ == "__main__":
    import glob

    image_paths = glob.glob("data/frames/obstacle_frames/*.png")
    print(f"Found {len(image_paths)} test images")

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        out, obs = detect_obstacles(frame)
        print(f"{path} → {len(obs)} obstacles")

        cv2.imshow("Obstacle Detection", out)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
