import cv2
import numpy as np

# --- AYARLAR ---
MIN_CONSECUTIVE_FRAMES = 8 
MAX_MISSED_FRAMES = 5
MAX_DISTANCE_THRESHOLD = 50 

# --- EK: MORFOLOJİK İŞLEM AYARLARI (Gölge Şekil Tespiti) ---
MORPH_KERNEL_SIZE = 19
MORPH_THRESHOLD = 20

class LaneCandidate:
    """
    Her bir potansiyel şerit parçasını temsil eden sınıf.
    Kendi geçmişini, konumunu ve güvenilirliğini takip eder.
    """
    def __init__(self, line, img_height):
        self.img_height = img_height
        self.line = line 
        self.found_count = 1
        self.missed_count = 0
        self.is_confirmed = False
        self.update_params(line)

    def update_params(self, line):
        x1, y1, x2, y2 = line
        if x2 == x1: return 
        self.line = line

    def is_similar_to(self, new_line):
        ox1, oy1, ox2, oy2 = self.line
        nx1, ny1, nx2, ny2 = new_line
        
        dist1 = np.sqrt((ox1 - nx1)**2 + (oy1 - ny1)**2)
        dist2 = np.sqrt((ox2 - nx2)**2 + (oy2 - ny2)**2)
        
        if dist1 < MAX_DISTANCE_THRESHOLD and dist2 < MAX_DISTANCE_THRESHOLD:
            return True
        return False

    def update(self, new_line):
        self.found_count += 1
        self.missed_count = 0
        
        # Yumuşatma (%70 Yeni, %30 Eski)
        self.line = [
            int(0.7 * new_line[0] + 0.3 * self.line[0]),
            int(0.7 * new_line[1] + 0.3 * self.line[1]),
            int(0.7 * new_line[2] + 0.3 * self.line[2]),
            int(0.7 * new_line[3] + 0.3 * self.line[3])
        ]
        
        if self.found_count >= MIN_CONSECUTIVE_FRAMES:
            self.is_confirmed = True

    def mark_missing(self):
        self.missed_count += 1
        if self.missed_count > MAX_MISSED_FRAMES:
            return False 
        return True 

# Global aday listesi
lane_candidates = []

def get_extended_line(line, img_height):
    x1, y1, x2, y2 = line
    if x2 == x1: return line 
    
    poly = np.polyfit([y1, y2], [x1, x2], 1)
    
    y_bottom = img_height
    y_top = int(img_height * 0.6)
    
    x_bottom = int(poly[0] * y_bottom + poly[1])
    x_top = int(poly[0] * y_top + poly[1])
    
    return [x_bottom, y_bottom, x_top, y_top]

def process_lanes_tracking(img, raw_lines):
    global lane_candidates
    height, width = img.shape[:2]
    
    if raw_lines is None:
        raw_lines = []
    
    # 1. Ham çizgileri uzat
    extended_lines = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1e-6: continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4: continue 
        
        ext_line = get_extended_line([x1, y1, x2, y2], height)
        extended_lines.append(ext_line)

    # 2. Eşleştirme
    matched_candidate_indices = set()
    
    for ext_line in extended_lines:
        found_match = False
        for i, candidate in enumerate(lane_candidates):
            if candidate.is_similar_to(ext_line):
                candidate.update(ext_line)
                matched_candidate_indices.add(i)
                found_match = True
                break 
        
        if not found_match:
            new_cand = LaneCandidate(ext_line, height)
            lane_candidates.append(new_cand)
            
    # 3. Temizlik
    candidates_to_keep = []
    for i, candidate in enumerate(lane_candidates):
        if i in matched_candidate_indices:
            candidates_to_keep.append(candidate)
        else:
            still_alive = candidate.mark_missing()
            if still_alive:
                candidates_to_keep.append(candidate)
    
    lane_candidates = candidates_to_keep

    # 4. Seçim ve Çizim
    confirmed_lanes = [c for c in lane_candidates if c.is_confirmed]
    final_left = None
    final_right = None
    center_x = width // 2
    
    best_left_candidates = []
    best_right_candidates = []
    
    for cand in confirmed_lanes:
        x_bottom = cand.line[0]
        if x_bottom < center_x:
            best_left_candidates.append(cand)
        else:
            best_right_candidates.append(cand)
            
    if best_left_candidates:
        final_left = max(best_left_candidates, key=lambda c: c.found_count).line
        
    if best_right_candidates:
        final_right = max(best_right_candidates, key=lambda c: c.found_count).line

    line_image = np.zeros_like(img)
    
    if final_left is not None:
        cv2.line(line_image, (final_left[0], final_left[1]), (final_left[2], final_left[3]), (255, 0, 0), 10)
    
    if final_right is not None:
        cv2.line(line_image, (final_right[0], final_right[1]), (final_right[2], final_right[3]), (0, 0, 255), 10)
        
    if final_left is not None and final_right is not None:
         pts = np.array([
            (final_left[0], final_left[1]),
            (final_left[2], final_left[3]),
            (final_right[2], final_right[3]),
            (final_right[0], final_right[1])
        ], np.int32)
         cv2.fillPoly(line_image, [pts], (0, 255, 0))

    return line_image

def detect_lanes(frame):
    height, width = frame.shape[:2]
    
    # ---------------------------------------------------------
    # BÖLÜM 1: RENK MASKELERİ (HLS/HSV)
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1.a) Beyaz Maskesi (Standart)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # 1.b) Sarı Maskesi (Standart)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 1.c) GÖLGE MASKESİ (SENİN BULDUĞUN DEĞERLER)
    # Değerler: H:108-130, S:45-52, V:187-197
    lower_shadow = np.array([108, 45, 187])
    upper_shadow = np.array([130, 52, 197])
    mask_shadow_color = cv2.inRange(hsv, lower_shadow, upper_shadow)
    
    # Renk maskelerini birleştir
    mask_color_combined = cv2.bitwise_or(mask_white, mask_yellow)
    mask_color_combined = cv2.bitwise_or(mask_color_combined, mask_shadow_color)
    
    # ---------------------------------------------------------
    # BÖLÜM 2: MORFOLOJİK İŞLEM (TOP-HAT)
    # Renkten bağımsız şekil tespiti için
    # ---------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Kernel: 19x19
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    tophat_img = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # Threshold: 20
    _, mask_morph = cv2.threshold(tophat_img, MORPH_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # ---------------------------------------------------------
    # BÖLÜM 3: BİRLEŞTİRME VE KENAR TESPİTİ
    # ---------------------------------------------------------
    
    # Renk Maskesi VEYA Morfoloji Maskesi
    final_mask = cv2.bitwise_or(mask_color_combined, mask_morph)
    
    # Maskeyi orijinal resme uygula (Sadece maskelenen yerler kalsın)
    masked_frame = cv2.bitwise_and(frame, frame, mask=final_mask)
    
    # Tekrar griye çevirip Canny uygula (Senin kodundaki akışa sadık kalıyoruz)
    masked_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # Hafif blur (Gürültü için)
    blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    
    # Canny Kenar Tespiti
    edges = cv2.Canny(blur, 50, 150)
    
    # ---------------------------------------------------------
    # BÖLÜM 4: ROI VE HOUGH
    # ---------------------------------------------------------
    
    # Senin kodundaki orijinal ROI
    roi_vertices = np.array([[
        (int(width * 0.05), int(height * 0.8)),
        (int(width * 0.2), int(height * 0.5)), 
        (int(width * 0.9), int(height * 0.5)), 
        (int(width * 0.95), int(height * 0.8))
    ]], dtype=np.int32)
    
    mask_roi = np.zeros_like(edges)
    cv2.fillPoly(mask_roi, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask_roi)
    
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,     
        minLineLength=20,
        maxLineGap=300
    )
    
    # ---------------------------------------------------------
    # BÖLÜM 5: TAKİP SİSTEMİ
    # ---------------------------------------------------------
    line_layer = process_lanes_tracking(frame, lines)
    
    # Birleştir
    result = cv2.addWeighted(frame, 1.0, line_layer, 0.4, 0)
    return result

# # --- ANA DÖNGÜ ---
# cap = cv2.VideoCapture('input.mp4') 

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
    
#     processed = process_frame(frame)
#     cv2.imshow('Smart Lane Tracking (Hybrid)', processed)
    
#     if cv2.waitKey(1) == 27: break

# cap.release()
# cv2.destroyAllWindows()