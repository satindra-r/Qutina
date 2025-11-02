import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import camera as QR

# Vertex shader
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out vec3 fragColor;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    fragColor = aColor;
}
"""

# Fragment shader
FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
"""

# Physical screen dimensions in cm
SCREEN_WIDTH_CM = float(input("Enter screen width (cm): "))
SCREEN_HEIGHT_CM = float(input("Enter screen height (cm): "))
FOCUS_RECT = input("Enter focus rect (y/n): ").startswith("y")

# Cube vertices (in cm, behind the screen)
points = [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1],
		  [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]]
points = [[p[0] * 5, p[1] * 5, p[2] * 5 + 50] for p in points]

points.extend([[-25, -5, 100], [25, -5, 100], [25, -5, 0], [-25, -5, 0]])

points = [[p[0], p[1] - 15, p[2]] for p in points]

faces = [[0, 1, 2, [1, 0, 0]], [0, 2, 3, [1, 0, 0]],
		 [1, 5, 6, [0, 1, 0]], [1, 6, 2, [0, 1, 0]],
		 [4, 0, 3, [0, 0, 1]], [4, 3, 7, [0, 0, 1]],
		 [3, 2, 6, [1, 1, 1]], [3, 6, 7, [1, 1, 1]],
		 [0, 1, 5, [1, 1, 0]], [0, 5, 4, [1, 1, 0]],
		 [5, 4, 7, [1, 0.5, 0]], [5, 6, 7, [1, 0.5, 0]],
		 [8, 9, 10, [0.1, 0.5, 1]], [8, 10, 11, [0.1, 0.5, 1]]]


def compile_shader_program():
	vertex_shader = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
	fragment_shader = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
	return compileProgram(vertex_shader, fragment_shader)


def dist_3d(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5


def project_off_axis(p, eye_pos, screen_width_cm, screen_height_cm):
	"""
	Off-axis perspective projection treating screen as a window.

	Args:
		p: 3D point in world space [x, y, z] (cm, with screen center at origin)
		eye_pos: Eye position [x, y, z] (cm, relative to screen center)
		screen_width_cm: Physical screen width in cm
		screen_height_cm: Physical screen height in cm

	Returns:
		[x, y] in normalized device coordinates [-1, 1]
	"""
	# Calculate relative position of point to eye
	rel_x = p[0] - eye_pos[0]
	rel_y = p[1] - eye_pos[1]
	rel_z = p[2] - eye_pos[2]

	# Point must be in front of eye (positive z from eye's perspective)
	# Since objects are at positive z and eye is at negative z, rel_z should be positive
	if rel_z <= 0.1:
		return None  # Point is behind or at eye position

	# Calculate where the ray from eye through point intersects screen plane (z=0)
	# Ray: eye_pos + t * (p - eye_pos), solve for z = 0
	# 0 = eye_pos[2] + t * (p[2] - eye_pos[2])
	# t = -eye_pos[2] / (p[2] - eye_pos[2])

	denom = p[2] - eye_pos[2]
	if abs(denom) < 0.001:
		return None

	t = -eye_pos[2] / denom

	if t < 0 or t > 1:
		return None  # Intersection is not between eye and point

	# Intersection point on screen plane
	screen_x = eye_pos[0] + t * (p[0] - eye_pos[0])
	screen_y = eye_pos[1] + t * (p[1] - eye_pos[1])

	# Convert to normalized device coordinates [-1, 1]
	# Screen extends from -width/2 to +width/2 in x, -height/2 to +height/2 in y
	ndc_x = (screen_x / (screen_width_cm / 2))
	ndc_y = (screen_y / (screen_height_cm / 2))

	# Clip if outside screen bounds (optional - can help with performance)
	if abs(ndc_x) > 3.0 or abs(ndc_y) > 3.0:
		return None

	return [ndc_x, ndc_y]


def same_side(p1, p2, p3, t1, t2):
	"""Check if points t1 and t2 are on the same side of plane defined by p1, p2, p3"""
	v1 = np.array([t1[0] - p1[0], t1[1] - p1[1], t1[2] - p1[2]])
	v2 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
	v3 = np.array([p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]])

	det1 = np.linalg.det(np.array([v1, v2, v3]))

	v1_2 = np.array([t2[0] - p1[0], t2[1] - p1[1], t2[2] - p1[2]])
	det2 = np.linalg.det(np.array([v1_2, v2, v3]))

	product = det1 * det2
	if abs(product) < 1e-10:
		return 0
	return 1 if product > 0 else -1


def is_behind(p11, p12, p13, p21, p22, p23, camera):
	"""
	Determine if face2 is behind face1 from camera view.
	Returns: 1 if face2 is behind face1, -1 if face1 is behind face2, 0 if unclear
	"""
	behind = [
		same_side(p11, p12, p13, p21, camera),
		same_side(p11, p12, p13, p22, camera),
		same_side(p11, p12, p13, p23, camera)
	]

	if max(behind) - min(behind) != 2:
		if max(behind) == 1:
			return 1
		else:
			return -1

	ahead = [
		same_side(p21, p22, p23, p11, camera),
		same_side(p21, p22, p23, p12, camera),
		same_side(p21, p22, p23, p13, camera)
	]

	if max(ahead) - min(ahead) != 2:
		if max(ahead) == 1:
			return -1
		else:
			return 1

	return 0


def sort_faces_painter(faces_list, points_list, camera):
	"""Sort faces using Painter's Algorithm (back to front for rendering)"""
	sorted_index = 1
	while sorted_index < len(faces_list):
		item = faces_list[sorted_index]
		insert_pos = sorted_index

		for i in range(sorted_index - 1, -1, -1):
			result = is_behind(
				points_list[faces_list[i][0]], points_list[faces_list[i][1]], points_list[faces_list[i][2]],
				points_list[item[0]], points_list[item[1]], points_list[item[2]],
				camera
			)

			# Reversed: if item is in front (result == -1), move it earlier
			if result == -1:
				insert_pos = i
			else:
				break

		if insert_pos < sorted_index:
			faces_list[insert_pos + 1:sorted_index + 1] = faces_list[insert_pos:sorted_index]
			faces_list[insert_pos] = item

		sorted_index += 1

	return faces_list


def create_triangle(p1, p2, p3, c):
	triangle = np.array([
		p1[0], p1[1], *c,
		p2[0], p2[1], *c,
		p3[0], p3[1], *c,
	], dtype=np.float32)
	return triangle


def main():
	global faces
	global points

	QR.init(SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM, FOCUS_RECT)
	pygame.init()

	info = pygame.display.Info()
	screen_width_px, screen_height_px = info.current_w, info.current_h

	pygame.display.set_mode((screen_width_px, screen_height_px),
							pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)

	shader_program = compile_shader_program()

	# Create VAO and VBO
	VAO = glGenVertexArrays(1)
	VBO = glGenBuffers(1)
	glBindVertexArray(VAO)
	glBindBuffer(GL_ARRAY_BUFFER, VBO)

	# Describe vertex layout
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
	glEnableVertexAttribArray(0)
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(8))
	glEnableVertexAttribArray(1)
	glBindVertexArray(0)

	clock = pygame.time.Clock()
	running = True

	while running:
		QR.main()

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False

		# Get eye position in cm relative to screen center
		eye_pos = QR.get_loc_3d()

		# Sort faces back-to-front for proper Painter's Algorithm rendering
		faces_sorted = sort_faces_painter(faces.copy(), points, eye_pos)

		# Project and render each face
		vertices = np.array([], dtype=np.float32)

		for face in faces_sorted:
			# Project the three vertices
			projected = []
			valid = True

			for i in range(3):
				proj = project_off_axis(points[face[i]], eye_pos, SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM)
				if proj is None:
					valid = False
					break
				projected.append(proj)

			# Only render if all vertices project successfully
			if valid:
				vertices = np.concatenate((vertices,
										   create_triangle(projected[0], projected[1], projected[2], face[3])))

		# Upload to GPU
		glBindBuffer(GL_ARRAY_BUFFER, VBO)
		glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

		# Render
		glClearColor(0.1, 0.1, 0.1, 1.0)
		glClear(GL_COLOR_BUFFER_BIT)

		glUseProgram(shader_program)
		glBindVertexArray(VAO)
		if vertices.size > 0:
			glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 5)
		glBindVertexArray(0)

		pygame.display.flip()
		clock.tick(60)

	glDeleteVertexArrays(1, [VAO])
	glDeleteBuffers(1, [VBO])
	glDeleteProgram(shader_program)
	pygame.quit()
	QR.deinit()


if __name__ == "__main__":
	main()
