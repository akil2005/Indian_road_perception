import cv2
import numpy as np
from modules.kalman_stabilizer import CarpetStabilizer
from modules.kalman_stabilizer import CentroidTracker

class SafetyManager:
    def __init__(self, fps=30.0):
        self.fps = fps
        self.stabilizer = CarpetStabilizer()
        self.ct = CentroidTracker()
        self.tunnel_poly = None
        self.object_data = {}
        
        # Global Alert State
        self.global_alert = "SAFE"
        self.global_color = (0, 255, 0)
        self.global_text = "PATH CLEAR"
        
        self.prev_target_x = None 
        
        # --- UI PERSISTENCE ---
        self.alert_timer = 0 
        self.last_threat_level = 0

    def generate_dynamic_carpet(self, da_mask, ll_mask, h, w):
        """
        UNIVERSAL LANE LOGIC: Supports both Left and Right lanes.
        Decides based on Lane Width, not hardcoded positions.
        """
        scan_y = int(h * 0.70)
        mid_x = w // 2
        
        # 1. Scan FULL Width (Remove the "Blinder")
        # We need to see the Right Shoulder if we are in the Right Lane!
        ll_slice = ll_mask[scan_y-5:scan_y+5, :]
        
        # Get all lane pixels
        all_pixels = np.where(ll_slice == 1)[1]
        
        # Default fallback
        target_x = mid_x 
        
        if len(all_pixels) == 0:
            # NO LINES? Drift slowly to previous position
            if self.prev_target_x is not None:
                target_x = self.prev_target_x
        else:
            # 2. Find "Left Candidate" and "Right Candidate" relative to center
            # L_cand = The pixel closest to center, but on the LEFT side
            # R_cand = The pixel closest to center, but on the RIGHT side
            
            l_candidates = all_pixels[all_pixels < mid_x]
            r_candidates = all_pixels[all_pixels >= mid_x]
            
            l_pos = int(l_candidates[-1]) if len(l_candidates) > 0 else None
            r_pos = int(r_candidates[0]) if len(r_candidates) > 0 else None
            
            # Standard Lane Width (approx 25% of screen)
            # Used for projection if one line is missing
            std_width = int(w * 0.23) 
            
            # Max Valid Width (approx 45% of screen)
            # If gap is bigger than this, we are looking at the opposite road edge!
            max_valid_width = int(w * 0.45)

            # --- DECISION LOGIC ---
            
            if l_pos is not None and r_pos is not None:
                # We see lines on BOTH sides. Check the gap.
                gap = r_pos - l_pos
                
                if gap < max_valid_width:
                    # Case A: VALID LANE (Highway Right Lane OR Highway Left Lane)
                    # The lines are close enough to be a single lane. Center strictly between them.
                    target_x = (l_pos + r_pos) // 2
                else:
                    # Case B: GAP TOO BIG (Two-Way Road or Missing Divider)
                    # We are seeing the Left Shoulder and the Far Right Edge.
                    # IGNORE r_pos. Trust l_pos (Left Edge) and project center.
                    target_x = l_pos + int(std_width * 0.5)
            
            elif l_pos is not None:
                # Case C: Only Left Line Visible
                # Project center to the right
                target_x = l_pos + int(std_width * 0.5)
                
            elif r_pos is not None:
                # Case D: Only Right Line Visible
                # Project center to the left
                target_x = r_pos - int(std_width * 0.5)

        # 3. Stabilization (Jump Rejection)
        if self.prev_target_x is not None:
            if abs(target_x - self.prev_target_x) > 60: # Allow bigger jumps for lane changes
                # Soft reject: Move heavily towards new target but don't snap
                target_x = int(0.6 * self.prev_target_x + 0.4 * target_x)
            else:
                target_x = int(0.7 * self.prev_target_x + 0.3 * target_x)
        
        self.prev_target_x = target_x

        # 4. Build Trapezoid
        hood_w = int(w * 0.29)
        top_w = int(w * 0.10)
        car_center = w // 2
        
        raw_poly = [
            [car_center - hood_w//2, h],      
            [target_x - top_w//2, scan_y],   
            [target_x + top_w//2, scan_y],    
            [car_center + hood_w//2, h]       
        ]
        
        self.tunnel_poly = self.stabilizer.stabilize(raw_poly)
        return self.tunnel_poly

    def check_hazard_status(self, detections_list, img_w, img_h):
        """
        Standard Hazard Logic (Same as before)
        """
        current_max_threat = 0
        
        rects = [d['box'] for d in detections_list]
        objects = self.ct.update(rects)
        results = []
        
        for det in detections_list:
            box = det['box']
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            h_curr = box[3] - box[1]
            
            obj_id = -1
            for oid, centroid in objects.items():
                if np.linalg.norm(np.array([cx, cy]) - np.array(centroid)) < 50:
                    obj_id = oid; break
            
            status = "SAFE"; color = (0, 255, 0)
            
            if obj_id != -1 and self.tunnel_poly is not None:
                dist_from_edge = cv2.pointPolygonTest(self.tunnel_poly, (cx, box[3]), True)
                
                if dist_from_edge < 15: 
                    self.object_data[obj_id] = {'h': h_curr, 'v_rel': 0}
                else:
                    prev = self.object_data.get(obj_id, {'h': h_curr, 'v_rel': 0})
                    v_rel_raw = h_curr - prev['h']
                    v_rel = (0.6 * v_rel_raw) + (0.4 * prev['v_rel'])
                    
                    if v_rel > 3.0: 
                        ttc = h_curr / (v_rel * self.fps)
                        if ttc < 1.5:
                            status = f"CRASH {ttc:.1f}s"; color = (0, 0, 255); current_max_threat = max(current_max_threat, 2)
                        elif ttc < 2.5:
                            status = f"WARN {ttc:.1f}s"; color = (0, 165, 255); current_max_threat = max(current_max_threat, 1)
                    
                    self.object_data[obj_id] = {'h': h_curr, 'v_rel': v_rel}
            
            results.append({'status': status, 'color': color})

        if current_max_threat > 0:
            self.alert_timer = 45 
            self.last_threat_level = current_max_threat
            if current_max_threat == 2:
                self.global_alert = "CRITICAL"; self.global_color = (0, 0, 255); self.global_text = "CRITICAL ALERT"
            elif current_max_threat == 1:
                self.global_alert = "CAUTION"; self.global_color = (0, 165, 255); self.global_text = "DISTANCE WARNING"
        elif self.alert_timer > 0:
            self.alert_timer -= 1
        else:
            self.global_alert = "SAFE"; self.global_color = (0, 255, 0); self.global_text = "PATH CLEAR"
            
        return results, self.global_color, self.global_text

    def draw_ui(self, frame, text, color):
        overlay = frame.copy()
        alpha = 0.4
        if self.tunnel_poly is not None: cv2.fillPoly(overlay, [self.tunnel_poly], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//2 - 150, h - 80), (w//2 + 150, h - 20), (0,0,0), -1)
        cv2.putText(frame, text, (w//2 - 120, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame
# import cv2
# import numpy as np
# from modules.kalman_stabilizer import CarpetStabilizer

# class CentroidTracker:
#     """
#     Assigns consistent IDs to objects based on distance.
#     Prevents ID switching when cars move slightly.
#     """
#     def __init__(self, max_disappeared=5):
#         self.nextObjectID = 0
#         self.objects = {} # ID -> Centroid (x, y)
#         self.disappeared = {} # ID -> Frames missing
#         self.max_disappeared = max_disappeared

#     def update(self, rects):
#         # Input: List of bounding box rects [(x1, y1, x2, y2), ...]
#         if len(rects) == 0:
#             for objectID in list(self.disappeared.keys()):
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)
#             return self.objects

#         inputCentroids = np.zeros((len(rects), 2), dtype="int")
#         for (i, (startX, startY, endX, endY)) in enumerate(rects):
#             cX = int((startX + endX) / 2.0)
#             cY = int((startY + endY) / 2.0)
#             inputCentroids[i] = (cX, cY)

#         if len(self.objects) == 0:
#             for i in range(0, len(inputCentroids)):
#                 self.register(inputCentroids[i])
#         else:
#             objectIDs = list(self.objects.keys())
#             objectCentroids = list(self.objects.values())
            
#             # Calculate distances between existing objects and new inputs
#             D = []
#             for i in range(len(objectCentroids)):
#                 row = []
#                 for j in range(len(inputCentroids)):
#                     dist = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))
#                     row.append(dist)
#                 D.append(row)
#             D = np.array(D)

#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]

#             usedRows = set()
#             usedCols = set()

#             for (row, col) in zip(rows, cols):
#                 if row in usedRows or col in usedCols: continue
#                 # If distance is too big, it's a different object
#                 if D[row, col] > 100: continue 

#                 objectID = objectIDs[row]
#                 self.objects[objectID] = inputCentroids[col]
#                 self.disappeared[objectID] = 0
#                 usedRows.add(row)
#                 usedCols.add(col)

#             unusedRows = set(range(0, D.shape[0])).difference(usedRows)
#             unusedCols = set(range(0, D.shape[1])).difference(usedCols)

#             for row in unusedRows:
#                 objectID = objectIDs[row]
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)

#             for col in unusedCols:
#                 self.register(inputCentroids[col])

#         return self.objects

#     def register(self, centroid):
#         self.objects[self.nextObjectID] = centroid
#         self.disappeared[self.nextObjectID] = 0
#         self.nextObjectID += 1

#     def deregister(self, objectID):
#         del self.objects[objectID]
#         del self.disappeared[objectID]

# class SafetyManager:
#     def __init__(self, fps=30.0):
#         self.fps = fps
#         self.stabilizer = CarpetStabilizer()
#         self.ct = CentroidTracker() # <--- NEW TRACKER
#         self.tunnel_poly = None
        
#         # Tracking Data: { id: {'h': height, 'cx': cx, 'v_rel': val} }
#         self.object_data = {}
        
#         # Global Alert State
#         self.global_alert = "SAFE"
#         self.global_color = (0, 255, 0)
#         self.global_text = "PATH CLEAR"
        
#         self.prev_target_x = None # For jump rejection

#     def generate_dynamic_carpet(self, da_mask, ll_mask, h, w):
#         """ Standard Stabilized Carpet Generation """
#         scan_y = int(h * 0.70)
#         mid_x = w // 2
        
#         ll_slice = ll_mask[scan_y-5:scan_y+5, :]
#         left_lanes = np.where(ll_slice[:, :mid_x] == 1)[1]
#         right_lanes = np.where(ll_slice[:, mid_x:] == 1)[1]
        
#         raw_target_x = w // 2
        
#         if len(left_lanes) > 0 and len(right_lanes) > 0:
#             raw_target_x = (int(np.median(left_lanes)) + int(np.median(right_lanes) + mid_x)) // 2
#         elif len(left_lanes) > 0:
#             raw_target_x = int(np.median(left_lanes)) + int(w * 0.18)
#         elif len(right_lanes) > 0:
#             raw_target_x = int(np.median(right_lanes) + mid_x) - int(w * 0.18)

#         # Jump Rejection
#         if self.prev_target_x is not None:
#             if abs(raw_target_x - self.prev_target_x) > 30:
#                 raw_target_x = self.prev_target_x
#             else:
#                 self.prev_target_x = raw_target_x
#         else:
#             self.prev_target_x = raw_target_x

#         hood_w = int(w * 0.29)
#         top_w = int(w * 0.10)
        
#         raw_poly = [
#             [w//2 - hood_w//2, h],
#             [raw_target_x - top_w//2, scan_y],
#             [raw_target_x + top_w//2, scan_y],
#             [w//2 + hood_w//2, h]
#         ]
        
#         self.tunnel_poly = self.stabilizer.stabilize(raw_poly)
#         return self.tunnel_poly

#     def check_hazard_status(self, detections_list, img_w, img_h):
#         """
#         Process all detections and assign INDIVIDUAL status.
#         Output: List of results matched to input list index.
#         """
#         self.global_alert = "SAFE"
#         self.global_color = (0, 255, 0)
#         self.global_text = "PATH CLEAR"
#         max_threat = 0
        
#         # 1. Update Tracker with new bounding boxes
#         rects = [d['box'] for d in detections_list]
#         objects = self.ct.update(rects) # Returns {ID: (cx, cy)}
        
#         results = []
        
#         # 2. Match Tracker IDs back to Input Detections
#         for det in detections_list:
#             box = det['box']
#             cx = int((box[0] + box[2]) / 2)
#             cy = int((box[1] + box[3]) / 2)
#             h_curr = box[3] - box[1]
            
#             # Find the ID for this box (by finding closest centroid)
#             obj_id = -1
#             min_dist = 9999
#             for oid, centroid in objects.items():
#                 d = np.linalg.norm(np.array([cx, cy]) - np.array(centroid))
#                 if d < 20: # Match found
#                     obj_id = oid
#                     min_dist = d
#                     break
            
#             # Default: Safe
#             status = "SAFE"
#             color = (0, 255, 0)
            
#             if obj_id != -1 and self.tunnel_poly is not None:
#                 # --- CHECK 1: CARPET CONSTRAINT ---
#                 # Check bottom center of box
#                 in_carpet = cv2.pointPolygonTest(self.tunnel_poly, (cx, box[3]), True)
                
#                 # Strict: Must be INSIDE (> 0) to matter. 
#                 # Exception: Head-On override disabled to stop false alarms per user request.
#                 if in_carpet < 0:
#                     status = "SAFE"
#                     color = (0, 255, 0) # Green box
                    
#                     # Still update velocity data for later
#                     self.object_data[obj_id] = {'h': h_curr, 'v_rel': 0}
                    
#                 else:
#                     # --- CHECK 2: VELOCITY LOGIC ---
#                     prev = self.object_data.get(obj_id, {'h': h_curr, 'v_rel': 0})
#                     v_rel_raw = h_curr - prev['h']
                    
#                     # Smooth it
#                     v_rel = (0.6 * v_rel_raw) + (0.4 * prev['v_rel'])
                    
#                     # Only warn if closing in (v_rel > 0)
#                     if v_rel > 1.0: 
#                         ttc = h_curr / (v_rel * self.fps) if v_rel > 0 else 99
                        
#                         if ttc < 1.5:
#                             status = f"CRASH {ttc:.1f}s"
#                             color = (0, 0, 255) # RED
#                             max_threat = max(max_threat, 2)
#                         elif ttc < 3.0:
#                             status = f"WARN {ttc:.1f}s"
#                             color = (0, 165, 255) # ORANGE
#                             max_threat = max(max_threat, 1)
                    
#                     # Save state
#                     self.object_data[obj_id] = {'h': h_curr, 'v_rel': v_rel}
            
#             results.append({'status': status, 'color': color})

#         # Set Global Carpet Status based on MAX threat found
#         if max_threat == 2:
#             self.global_alert = "CRITICAL"
#             self.global_color = (0, 0, 255)
#             self.global_text = "CRITICAL ALERT"
#         elif max_threat == 1:
#             self.global_alert = "CAUTION"
#             self.global_color = (0, 165, 255)
#             self.global_text = "DISTANCE WARNING"
            
#         return results, self.global_color, self.global_text

#     def draw_ui(self, frame, text, color):
#         overlay = frame.copy()
#         alpha = 0.4
#         if self.tunnel_poly is not None: cv2.fillPoly(overlay, [self.tunnel_poly], color)
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
#         h, w = frame.shape[:2]
#         cv2.rectangle(frame, (w//2 - 150, h - 80), (w//2 + 150, h - 20), (0,0,0), -1)
#         cv2.putText(frame, text, (w//2 - 120, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         return frame