import cv2
import numpy as np

# =========================================
# PART 1: THE FLICKER KILLER (Kalman Filter)
# =========================================

class PointKalman:
    """
    Stabilizes a single 2D point (x,y) using a Constant Velocity Model.
    """
    def __init__(self, responsiveness=0.03):
        # 4 state variables (x, y, dx, dy), 2 measured (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        
        # --- TUNING KNOB ---
        # responsiveness: Higher = Snappier, Lower = Smoother
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * responsiveness
        
        # Measurement Noise (Fixed trust in AI input)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        self.first_run = True

    def update(self, x, y):
        if self.first_run:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.first_run = False
            return x, y

        self.kf.predict()
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        
        predicted_x = int(self.kf.statePost[0, 0])
        predicted_y = int(self.kf.statePost[1, 0])
        
        return predicted_x, predicted_y

class CarpetStabilizer:
    """
    Manages 4 Kalman filters for the Trapezoid Corners.
    """
    def __init__(self):
        # We set responsiveness=0.01 for a very heavy, stable carpet.
        # If it feels too slow on curves, change this to 0.05
        self.filters = [PointKalman(responsiveness=0.01) for _ in range(4)]

    def stabilize(self, poly_points):
        stabilized = []
        for i, pt in enumerate(poly_points):
            sx, sy = self.filters[i].update(pt[0], pt[1])
            stabilized.append([sx, sy])
        return np.array(stabilized, np.int32)


# =========================================
# PART 2: THE ID KEEPER (Centroid Tracker)
# =========================================

class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.nextObjectID = 0
        self.objects = {} 
        self.disappeared = {} 
        self.max_disappeared = max_disappeared

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            D = []
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                if D[row, col] > 100: continue 

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]