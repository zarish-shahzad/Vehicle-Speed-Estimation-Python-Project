# ============================================================
# VEHICLE SPEED ESTIMATION USING YOLO + BUILT-IN TRACKER
# ============================================================
# STEP 1  : Import required libraries
# STEP 2  : Define user settings (video, lines, real distance)
# STEP 3  : Define colors for visualization
# STEP 4  : Load YOLO model with built-in tracking
# STEP 5  : Initialize video capture and FPS
# STEP 6  : Initialize tracking data structures
# STEP 7  : Read video frame-by-frame
# STEP 8  : Perform YOLO detection + tracking
# STEP 9  : Draw reference lines (Line A & Line B)
# STEP 10 : Filter vehicle classes only
# STEP 11 : Calculate object center point
# STEP 12 : Detect line crossing events
# STEP 13 : Store frame numbers for crossings
# STEP 14 : Compute speed using distance & time
# STEP 15 : Draw bounding boxes, IDs, speed labels
# STEP 16 : Display output frame
# STEP 17 : Exit and cleanup
# ============================================================


# ----------------------------
# STEP 1 : IMPORT LIBRARIES
# ----------------------------
import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------
# STEP 2 : USER SETTINGS
# ----------------------------
VIDEO_PATH = "01.mp4"

# Two horizontal reference lines
first_line  = [(344,324), (800,324)]    # Line A
second_line = [(300,381), (850,381)]    # Line B

REAL_DISTANCE_M = 8     # Real-world distance between lines (meters)
MARGIN = 8              # Pixel tolerance for line crossing

# ----------------------------
# STEP 3 : COLORS (BRIGHT)
# ----------------------------
LINE_COLOR   = (0, 255, 255)    # Neon Yellow
BOX_COLOR    = (0, 255, 0)      # Neon Green
TEXT_COLOR   = (255, 0, 0)    # Bright Yellow
CENTER_COLOR = (255, 255, 0)    # Pink


# ----------------------------
# STEP 4 : LOAD YOLO MODEL
# ----------------------------
# YOLO 11 with built-in tracker
model = YOLO("yolo11x.pt")


# ----------------------------
# STEP 5 : VIDEO CAPTURE SETUP
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc,fps,(1280,720))

# ----------------------------
# STEP 6 : INITIALIZE VARIABLES
# ----------------------------
frame_id = 0

# Dictionary format:
# { track_id : {"A": frame_no, "B": frame_no, "speed": value} }
cross_time = {}

# Extract Y-coordinates of lines
LINE_Y1 = first_line[0][1]
LINE_Y2 = second_line[0][1]

# ============================================================
# STEP 7 : MAIN VIDEO LOOP
# ============================================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # --------------------------------------------------------
    # STEP 8 : YOLO DETECTION + TRACKING
    # --------------------------------------------------------
    results = model.track(
                # The current video frame on which detection + tracking is performed
                frame,

                # persist=True:
                # Keeps object IDs consistent across frames
                # (so the same vehicle keeps the same ID while moving)
                persist=True,

                # verbose=False:
                # Disables console logs for every frame
                # (keeps terminal clean and faster)
                verbose=False
    )[0]


    # --------------------------------------------------------
    # STEP 9 : DRAW REFERENCE LINES
    # --------------------------------------------------------
    cv2.line(frame, first_line[0], first_line[1], LINE_COLOR, 3)
    cv2.putText(frame, "Line A",
                (first_line[0][0], LINE_Y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                LINE_COLOR, 2)

    cv2.line(frame, second_line[0], second_line[1], LINE_COLOR, 3)
    cv2.putText(frame, "Line B",
                (second_line[0][0], LINE_Y2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                LINE_COLOR, 2)


    # --------------------------------------------------------
    # STEP 10 : PROCESS EACH TRACKED OBJECT
    # --------------------------------------------------------
    if results.boxes is not None:

        for box in results.boxes:

            # STEP 10.1 : FILTER VEHICLE CLASSES
            cls = int(box.cls[0])

            # COCO IDs: car=2, motorcycle=3, bus=5, truck=7
            if cls not in [2, 3, 5, 7]:
                continue


            # STEP 10.2 : BOUNDING BOX
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            # STEP 10.3 : TRACKING ID
            if box.id is None:
                continue

            track_id = int(box.id[0])


            # STEP 11 : CENTER POINT
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2


            # STEP 12 : INITIALIZE TRACK ENTRY
            if track_id not in cross_time:
                cross_time[track_id] = {
                    "A": None,
                    "B": None,
                    "speed": None
                }


            # ------------------------------------------------
            # STEP 13 : LINE CROSSING DETECTION
            # ------------------------------------------------

            # ---- Line A ----
            # cross_time[3]["A"] = None  → object ID 3 has not crossed Line A before
            if cross_time[track_id]["A"] is None:

                # Define the crossing tolerance range around Line A
                # Suppose:
                # LINE_Y1 = 300   → Y-position of Line A
                # MARGIN  = 8     → allowed error margin
                #
                # Lower bound = LINE_Y1 - MARGIN = 300 - 8 = 292
                # Upper bound = LINE_Y1 + MARGIN = 300 + 8 = 308
                #
                # So valid crossing range becomes: 292 ≤ cy ≤ 308
                if (LINE_Y1 - MARGIN) <= cy <= (LINE_Y1 + MARGIN):
                    cross_time[track_id]["A"] = frame_id

            # ---- Line B ----
            if cross_time[track_id]["A"] is not None and cross_time[track_id]["B"] is None:

                if (LINE_Y2 - MARGIN) <= cy <= (LINE_Y2 + MARGIN):
                    cross_time[track_id]["B"] = frame_id

                    # STEP 14 : SPEED CALCULATION
                    f1 = cross_time[track_id]["A"]
                    f2 = cross_time[track_id]["B"]

                    time_s = (f2 - f1) / fps
                    if time_s > 0:
                        speed = (REAL_DISTANCE_M / time_s) * 3.6
                        cross_time[track_id]["speed"] = speed


            # ------------------------------------------------
            # STEP 15 : DRAW RESULTS
            # ------------------------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)

            label = f"ID {track_id}"
            
            color =TEXT_COLOR
            if cross_time[track_id]["speed"] is not None:
                label = f"{cross_time[track_id]['speed']:.1f} km/h"
                color = (0,0,255)

            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            cv2.circle(frame, (cx, cy), 6, CENTER_COLOR, -1)
            color = TEXT_COLOR


    # --------------------------------------------------------
    # STEP 16 : DISPLAY OUTPUT
    # --------------------------------------------------------
    cv2.imshow("YOLO Vehicle Speed Estimation", frame)
    out.write(frame)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break


# ============================================================
# STEP 17 : CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
