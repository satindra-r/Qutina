import cv2
import mediapipe as mp
import numpy as np

# ----------------------------
# Setup constants and globals
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

screen_width = 30.4
screen_height = 19

FOV = 3 / 4
PIXEL_CONV = 800

MAX_ERR = 1.5
LERP = 0.75

loc_3d_smoothed = [0, 0, 0]
loc_3d = [0, 0, 0]
loc_2d = [0, 0]
loc_1d = 0


# ----------------------------
# Utility functions
# ----------------------------
def dist(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def point_in_polygon(point, polygon):
	"""Check if a point is inside a polygon using cv2.pointPolygonTest"""
	# Convert point to tuple of regular Python floats/ints
	point = (float(point[0]), float(point[1]))
	return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0


def get_loc_3d():
	global loc_3d_smoothed
	return loc_3d_smoothed


def get_rect_roi(rect, frame_shape, padding):
	"""Get bounding box ROI from rectangle with padding"""
	h, w = frame_shape[:2]
	points = rect.reshape(-1, 2)
	x_min = max(0, int(np.min(points[:, 0])) - padding)
	x_max = min(w, int(np.max(points[:, 0])) + padding)
	y_min = max(0, int(np.min(points[:, 1])) - padding)
	y_max = min(h, int(np.max(points[:, 1])) + padding)
	return x_min, y_min, x_max, y_max


def find_largest_rectangle(frame):
	"""Detects the largest rectangular contour (within area limits) and returns its corner points."""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blur, 50, 150)

	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	height, width = frame.shape[:2]
	frame_area = width * height

	min_area_ratio = 0.001  # ignore too small rectangles
	max_area_ratio = 0.3  # ignore overly large rectangles
	min_area = frame_area * min_area_ratio
	max_area = frame_area * max_area_ratio

	max_area_found = 0
	best_rect = None

	for cnt in contours:
		epsilon = 0.02 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)

		if len(approx) == 4 and cv2.isContourConvex(approx):
			area = cv2.contourArea(approx)
			if min_area < area < max_area:
				if area > max_area_found:
					max_area_found = area
					best_rect = approx

	return best_rect


# ----------------------------
# Main processing functions
# ----------------------------
def init(width, height, rectangle_mode=True):
	global cap, face_mesh, screen_width, screen_height, FOCUS_RECT

	screen_width = width
	screen_height = height

	FOCUS_RECT = rectangle_mode

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Error: Could not open webcam")
		exit(0)
	face_mesh = mp_face_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	)


def main():
	global face_mesh, loc_3d_smoothed, loc_3d, loc_2d, loc_1d, FOCUS_RECT

	ret, frame = cap.read()
	if not ret:
		print("Error: Can't receive frame from webcam")
		return True

	h, w, _ = frame.shape

	# Only detect rectangle if feature is enabled
	rect = None
	if FOCUS_RECT:
		rect = find_largest_rectangle(frame)

		# Determine ROI for face detection
		if rect is not None:
			# Draw the detected rectangle
			for i in range(4):
				pt1 = tuple(rect[i][0])
				pt2 = tuple(rect[(i + 1) % 4][0])
				cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

			# Get ROI around rectangle
			x_min, y_min, x_max, y_max = get_rect_roi(rect, frame.shape, 150)
			roi_frame = frame[y_min:y_max, x_min:x_max]

			# Draw ROI boundary
			cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

			# Convert ROI to RGB for face detection
			rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
			roi_offset = (x_min, y_min)
		else:
			# No rectangle - use full frame
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			roi_offset = (0, 0)
	else:
		# Rectangle feature disabled - always use full frame
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		roi_offset = (0, 0)

	# Face + iris detection
	results = face_mesh.process(rgb)

	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			# Adjust coordinates based on ROI offset
			left_eye_points = np.array([(int(face_landmarks.landmark[i].x * (rgb.shape[1])) + roi_offset[0],
										 int(face_landmarks.landmark[i].y * (rgb.shape[0])) + roi_offset[1])
										for i in LEFT_IRIS])
			right_eye_points = np.array([(int(face_landmarks.landmark[i].x * (rgb.shape[1])) + roi_offset[0],
										  int(face_landmarks.landmark[i].y * (rgb.shape[0])) + roi_offset[1])
										 for i in RIGHT_IRIS])

			# Calculate iris centers as the mean of all 4 boundary points
			left_center = np.mean(left_eye_points, axis=0).astype(int)
			right_center = np.mean(right_eye_points, axis=0).astype(int)

			# Draw iris outlines
			for i in range(len(left_eye_points)):
				pt1 = tuple(left_eye_points[i])
				pt2 = tuple(left_eye_points[(i + 1) % len(left_eye_points)])
				cv2.line(frame, pt1, pt2, (50, 150, 50), 3)

			for i in range(len(right_eye_points)):
				pt1 = tuple(right_eye_points[i])
				pt2 = tuple(right_eye_points[(i + 1) % len(right_eye_points)])
				cv2.line(frame, pt1, pt2, (50, 150, 50), 3)

			# Draw iris centers
			cv2.circle(frame, tuple(left_center), 4, (255, 0, 0), -1)
			cv2.circle(frame, tuple(right_center), 4, (0, 0, 255), -1)

			# Determine if we should update position based on rectangle mode
			if FOCUS_RECT:
				# Rectangle mode: check if right eye is inside rectangle
				right_eye_inside = rect is not None and point_in_polygon(tuple(right_center), rect.reshape(-1, 2))
				should_update = rect is None or right_eye_inside

				# Show status
				if rect is None:
					cv2.putText(frame, "No Rectangle - Tracking Active", (30, 50),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
				elif right_eye_inside:
					cv2.putText(frame, "Right Eye INSIDE Rectangle", (30, 50),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
				else:
					cv2.putText(frame, "Right Eye OUTSIDE Rectangle", (30, 50),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
			else:
				# No rectangle mode: always track
				should_update = True
				cv2.putText(frame, "Eye Tracking (Rectangle OFF)", (30, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

			# Only proceed with 3D localization if we should update
			if should_update:
				# 3D localization
				relative_eye_center = [(left_center[0] + right_center[0]) / 2 - w / 2,
									   (left_center[1] + right_center[1]) / 2 - h / 2]
				cv2.line(frame, [w // 2, h // 2],
						 [w // 2 + int(relative_eye_center[0]), h // 2 + int(relative_eye_center[1])],
						 (0, 0, 255), 3)

				relative_eye_dist = dist(relative_eye_center, [0, 0])
				left_avg_side = float(
					np.mean([dist(left_eye_points[i], left_eye_points[(i + 1) % 4]) for i in range(4)]))

				loc_1d = PIXEL_CONV / left_avg_side
				loc_2d = [loc_1d * (FOV * relative_eye_dist) / ((w ** 2 + (FOV * relative_eye_dist) ** 2) ** 0.5),
						  -loc_1d * w / ((w ** 2 + (FOV * relative_eye_dist) ** 2) ** 0.5)]
				loc_3d = [-loc_2d[0] * relative_eye_center[0] / relative_eye_dist,
						  -loc_2d[0] * relative_eye_center[1] / relative_eye_dist, loc_2d[1]]
				loc_3d[2] -= screen_height / 2

				# Calculate change from last smoothed position
				del_loc_3d = [loc_3d[i] - loc_3d_smoothed[i] for i in range(3)]
				magnitude = sum([d ** 2 for d in del_loc_3d]) ** 0.5

				# Apply smoothing when error is SMALL (normal movement)
				# Skip smoothing when error is LARGE (erratic/noise)
				if magnitude > MAX_ERR:
					loc_3d_smoothed = [LERP * loc_3d_smoothed[i] + (1 - LERP) * loc_3d[i] for i in range(3)]

	cv2.imshow('Eye + Rectangle Detection', frame)
	cv2.waitKey(1)
	return True


def deinit():
	cap.release()
	cv2.destroyAllWindows()


# ----------------------------
# Main loop
# ----------------------------
if __name__ == '__main__':
	# Initialize with rectangle mode enabled (change to False to disable)
	init(13.4, 19, True)

	while True:
		if not main():
			break
		print([int(f) for f in get_loc_3d()])
	deinit()
