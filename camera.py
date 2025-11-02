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
def init(width, height):
	global cap, face_mesh, initialized, screen_width, screen_height

	screen_width = width
	screen_height = height

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Error: Could not open webcam.")
		exit()
	face_mesh = mp_face_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	)
	initialized = False


def main():
	global face_mesh, loc_3d_smoothed, loc_3d, loc_2d, loc_1d, initialized

	ret, frame = cap.read()
	if not ret:
		print("Error: Can't receive frame (stream end?). Exiting ...")
		return False

	# Detect rectangle in the frame
	rect = find_largest_rectangle(frame)

	if rect is not None:
		# Draw the detected rectangle
		for i in range(4):
			pt1 = tuple(rect[i][0])
			pt2 = tuple(rect[(i + 1) % 4][0])
			cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

	# Face + iris detection
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(rgb)

	if results.multi_face_landmarks:
		h, w, _ = frame.shape
		for face_landmarks in results.multi_face_landmarks:
			left_eye_points = np.array([(int(face_landmarks.landmark[i].x * w),
										 int(face_landmarks.landmark[i].y * h))
										for i in LEFT_IRIS])
			right_eye_points = np.array([(int(face_landmarks.landmark[i].x * w),
										  int(face_landmarks.landmark[i].y * h))
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

			# Check if RIGHT eye is inside rectangle (only if rectangle exists)
			right_eye_inside = rect is not None and point_in_polygon(tuple(right_center), rect.reshape(-1, 2))

			# Determine if we should update position
			should_update = rect is None or right_eye_inside

			# Show status
			if rect is None:
				cv2.putText(frame, "No Rectangle - Tracking Active", (30, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
			elif right_eye_inside:
				cv2.putText(frame, "Right Eye INSIDE Rectangle", (30, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
			else:
				cv2.putText(frame, "Right Eye OUTSIDE Rectangle", (30, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

			# Only proceed with 3D localization if we should update
			if should_update:
				# 3D localization
				relative_eye_center = [(left_center[0] + right_center[0]) / 2 - w / 2,
									   (left_center[1] + right_center[1]) / 2 - h / 2]
				cv2.line(frame, [w // 2, h // 2],
						 [w // 2 + int(relative_eye_center[0]), h // 2 + int(relative_eye_center[1])],
						 (0, 0, 255), 3)

				relative_eye_dist = dist(relative_eye_center, [0, 0])
				left_avg_side = np.mean([dist(left_eye_points[i], left_eye_points[(i + 1) % 4]) for i in range(4)])

				loc_1d = PIXEL_CONV / left_avg_side
				loc_2d = [loc_1d * (FOV * relative_eye_dist) / ((w ** 2 + (FOV * relative_eye_dist) ** 2) ** 0.5),
						  -loc_1d * w / ((w ** 2 + (FOV * relative_eye_dist) ** 2) ** 0.5)]
				loc_3d = [-loc_2d[0] * relative_eye_center[0] / relative_eye_dist,
						  -loc_2d[0] * relative_eye_center[1] / relative_eye_dist, loc_2d[1]]
				loc_3d[2] -= screen_height / 2

				# Calculate change from last smoothed position
				del_loc_3d = [loc_3d[i] - loc_3d_smoothed[i] for i in range(3)]
				magnitude = sum([d ** 2 for d in del_loc_3d]) ** 0.5

				if magnitude > MAX_ERR:
					loc_3d_smoothed = [LERP * loc_3d_smoothed[i] + (1 - LERP) * loc_3d[i] for i in range(3)]
			# If magnitude > MAX_ERR, keep the previous smoothed value

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
	init()
	while main():
		print([int(f) for f in get_loc_3d()])
	deinit()
