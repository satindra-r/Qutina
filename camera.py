import cv2

FOV = 3 / 4
PIXEL_CONV = 3500


def dist(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# Initialize webcam
# cap = cv2.VideoCapture(int(input('Enter Camera ID: ')))

cap = cv2.VideoCapture(2)

if not cap.isOpened():
	print("Error: Could not open webcam.")
	exit()

# Initialize QR code detector
detector = cv2.QRCodeDetector()

while True:
	ret, frame = cap.read()
	if not ret:
		print("Error: Can't receive frame (stream end?). Exiting ...")
		break

	try:
		# Detect and decode QR code
		data, points, _ = detector.detectAndDecode(frame)
	except cv2.error as e:
		continue;

	if points is not None:
		loc_3d = [0, 0, 0];
		loc_2d = [0, 0];
		loc_1d = 0;
		# Convert corner points to integer for drawing
		points = points[0].astype(int)
		# Draw a bounding box around the detected QR code
		for i in range(len(points)):
			pt1 = tuple(points[i])
			pt2 = tuple(points[(i + 1) % len(points)])
			cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

		avg_side = (dist(points[0], points[1]) + dist(points[1], points[2]) + dist(points[2], points[3]) + dist(
			points[3], points[1])) / 4

		height, width, channels = frame.shape

		avg_loc = ((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4 - width / 2,
				   (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4 - height / 2)

		avg_loc_dist = dist(avg_loc, [0, 0])

		#cv2.line(frame, [width // 2, height // 2], [width // 2 + int(avg_loc[0]), height // 2 + int(avg_loc[1])],(255, 0, 0), 3)

		loc_1d = PIXEL_CONV / avg_side
		loc_2d = [loc_1d * (FOV * avg_loc_dist) / ((width ** 2 + (FOV * avg_loc_dist) ** 2) ** 0.5),
				  loc_1d * width / ((width ** 2 + (FOV * avg_loc_dist) ** 2) ** 0.5)]
		loc_3d = [-loc_2d[0] * avg_loc[0] / avg_loc_dist, loc_2d[0] * avg_loc[1] / avg_loc_dist, loc_2d[1]]

		print([int(f) for f in loc_3d])

		# Display decoded text above the QR code
		if data:
			x, y = points[0]
			cv2.putText(frame, data, (x, y - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

	# Show the webcam feed
	cv2.imshow('QR Code Detection', frame)

	# Quit if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release and close
cap.release()
cv2.destroyAllWindows()
