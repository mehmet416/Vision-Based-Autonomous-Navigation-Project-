import cv2
import numpy as np

def detect_barriers(frame):


    # --- SETTINGS ---
    # 1. ROI (Region of Interest) - Where the barrier sits when closed
    # Read the first frame to get dimensions
    h, w = frame.shape[:2]
    
    # Define ROI coordinates (Adjust these percentages based on your specific video)
    roi_y1, roi_y2 = int(h * 0.3), int(h * 0.4)
    roi_x1, roi_x2 = int(w * 0.20), int(w * 0.85)

    # 2. Threshold Values
    # Minimum number of pixels required to consider the feature "present"
    MIN_EDGE_PIXELS = 200    # Is there a long enough horizontal line?
    MIN_RED_PIXELS = 50      # Are there red reflectors inside that line?
    MIN_WHITE_PIXELS = 50    # Is there a white body inside that line?

    print("Hybrid Analysis Started...")
    print("Logic: Horizontal Edge && Red Color && White Color -> CLOSED")


        # Crop ROI
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # -----------------------------------------------------------
        # STEP 1: SOBEL Y (Find Horizontal Structure)
        # -----------------------------------------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Sobel Y derivative (Detects only horizontal changes)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
        
        # Normalize to 0-255 and convert to uint8
    sobel_8u = np.uint8(255 * sobel_y / np.max(sobel_y + 1e-9))
        
        # Threshold: Keep only strong horizontal edges
    _, edge_mask = cv2.threshold(sobel_8u, 50, 255, cv2.THRESH_BINARY)
        
        # Dilation: 
    # Edges are thin. We dilate the mask to cover the "body" of the barrier
    # so we can check for colors inside it. Using a horizontal kernel.
    kernel_dilate = np.ones((5, 15), np.uint8) 
    structure_mask = cv2.dilate(edge_mask, kernel_dilate, iterations=1)
    
    # Structure Score (How much horizontal object is there?)
    structure_score = cv2.countNonZero(structure_mask)

    # -----------------------------------------------------------
    # STEP 2: COLOR ANALYSIS (Red and White)
    # -----------------------------------------------------------
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red Mask (For Reflectors)
    # Red wraps around 0/180 in HSV, so we need two ranges.
    mask_r1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_r1, mask_r2)
    
    # White Mask (For Barrier Body)
    # Low Saturation, High Value (Value lowered slightly to catch shadows)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))

    # -----------------------------------------------------------
    # STEP 3: FUSION
    # We look for colors ONLY inside the "Structure Mask".
    # This prevents background objects (red cars, white buildings) from triggering detection.
    # -----------------------------------------------------------
    
    # Red pixels overlapping with horizontal edges
    red_in_barrier = cv2.bitwise_and(red_mask, red_mask, mask=structure_mask)
    red_score = cv2.countNonZero(red_in_barrier)
    
    # White pixels overlapping with horizontal edges
    white_in_barrier = cv2.bitwise_and(white_mask, white_mask, mask=structure_mask)
    white_score = cv2.countNonZero(white_in_barrier)

    # -----------------------------------------------------------
    # STEP 4: DECISION MECHANISM
    # -----------------------------------------------------------
    
    # Condition: Is there Structure? AND Red? AND White?
    is_structure_ok = structure_score > MIN_EDGE_PIXELS
    is_red_ok = red_score > MIN_RED_PIXELS
    is_white_ok = white_score > MIN_WHITE_PIXELS
    
    if is_structure_ok and is_red_ok and is_white_ok:
        status = "CLOSED (BARRIER DETECTED)"
        color_status = (0, 0, 255) # Red
        
        # Draw a bounding box around the detected structure
        contours, _ = cv2.findContours(structure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500: # Ignore small noise
                x, y, w, h = cv2.boundingRect(cnt)
                # Translate ROI coordinates to Frame coordinates
                cv2.rectangle(frame, (roi_x1 + x, roi_y1 + y), (roi_x1 + x + w, roi_y1 + y + h), (0, 0, 255), 2)
    else:
        status = "OPEN (PASSAGE FREE)"
        color_status = (0, 255, 0) # Green

    # -----------------------------------------------------------
    # VISUALIZATION PANEL
        # -----------------------------------------------------------
        
        # 1. Structure Mask (Visualized as Blue)
    vis_structure = cv2.cvtColor(structure_mask, cv2.COLOR_GRAY2BGR)
    vis_structure[:, :, 0] = 255 # Boost Blue channel
    vis_structure[:, :, 1] = 0
    vis_structure[:, :, 2] = 0
    
    # 2. Color Masks (Visualized as Red and White)
    vis_red = cv2.cvtColor(red_in_barrier, cv2.COLOR_GRAY2BGR)
    vis_red[:, :, 2] = 255 # Red only
    
    vis_white = cv2.cvtColor(white_in_barrier, cv2.COLOR_GRAY2BGR) # White stays white
    
    # Blend them all into one analysis image
    # Structure (Base) + Red + White
    analysis_view = cv2.addWeighted(vis_structure, 0.3, vis_red, 1.0, 0)
    analysis_view = cv2.addWeighted(analysis_view, 1.0, vis_white, 1.0, 0)
    
    # Add Debug Text to the small panel
    cv2.putText(analysis_view, f"Struct: {structure_score}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(analysis_view, f"Red: {red_score}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(analysis_view, f"White: {white_score}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Resize panel for display
    analysis_view_large = cv2.resize(analysis_view, (400, 150))
    
    # Draw ROI Box and Status on Main Frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), color_status, 2)
    cv2.putText(frame, status, (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
    
    # Attach Analysis Panel to the right side of the frame
    final_h = frame.shape[0]
    
    side_panel = np.zeros((final_h, 400, 3), dtype=np.uint8)
    # Center the analysis view vertically
    y_offset = (final_h - 150) // 2
    side_panel[y_offset:y_offset+150, :] = analysis_view_large
    
    # Explanatory Text on Side Panel
    cv2.putText(side_panel, "ANALYSIS DETAIL", (20, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(side_panel, "Blue: Sobel (Horizontal)", (20, y_offset + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(side_panel, "Red: Reflector", (20, y_offset + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(side_panel, "White: Body", (20, y_offset + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    final_frame = np.hstack((frame, side_panel))

    cv2.imshow("Hybrid Barrier Detection", final_frame)


    return frame