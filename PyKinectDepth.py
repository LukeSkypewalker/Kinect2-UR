import threading
from time import sleep, time

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import ctypes
import pygame
import numpy as np
import urx
from coord_transformation import get_transform
from kinect_point_generator import PointGenerator

frame_width = 512
frame_height = 424
frame_width_half = frame_width / 2
frame_height_half = frame_height / 2
fov_horiz = np.deg2rad(70.6)
fov_vert = np.deg2rad(60)
fov_horiz_half = fov_horiz / 2
fov_vert_half = fov_vert / 2
T1 = []
T2 = []
T3 = []


def find_closest_element_index(frame):
    valid_idx = np.where(frame > 0)[0]
    return valid_idx[frame[valid_idx].argmin()]


def find_top_point_index(frame):
    unit_frame = frame.copy()
    unit_frame[unit_frame > 0] = 1
    return unit_frame.argmax()


def noise_filter(last_frames):
    mask = np.all(last_frames > 13, 0)
    filtered_frame = np.copy(last_frames[0])
    filtered_frame[~mask] = 0
    return filtered_frame


def enqueue(item, array):
    result = np.roll(array, 1, 0)
    result[0] = item
    return result


def calc_coords_in_mm(x_in_pixels, y_in_pixels, depth):
    x_half = x_in_pixels - frame_width_half
    y_half = -(y_in_pixels - frame_height_half)
    x_in_mm = depth * np.math.tan(fov_horiz_half) * (x_half / frame_width_half)
    y_in_mm = depth * np.math.tan(fov_vert_half) * (y_half / frame_height_half)
    return x_in_mm, y_in_mm, depth


def calc_coords_in_mm2(x_in_pixels, y_in_pixels, depth):
    cx = 254.878
    cy = 205.395
    fx = 365.456
    fy = 365.456

    z = depth/1000
    x = (x_in_pixels - cx) * z / fx
    y = (y_in_pixels - cy) * z / fy
    return x, y, z


def calc_coords_in_mm3(x_in_pixels, y_in_pixels, depth):
    cx = 2.5246874698519187e+02
    cy = 2.0724109283529990e+02
    fx = 3.6603389564721579e+02
    fy = 3.6671567662701682e+02
    k1 = 1.0256662997128010e-01
    k2 = -3.0267317007504330e-01
    k3 = -1.3424675613181905e-03
    p1 = -9.1787629334898115e-04
    p2 = 1.2367949702286907e-01

    R = x_in_pixels**2 + y_in_pixels**2
    K = 1 + k1*R + k2*R**2 + k3*R**3
    x_px_corrected = x_in_pixels*K + 2*p1*x_in_pixels*y_in_pixels + p2*(R+2*x_in_pixels**2)
    y_px_corrected = y_in_pixels*K + p1*(R+2*y_in_pixels**2) + 2*p2*x_in_pixels*y_in_pixels

    z = depth/1000
    x = (x_px_corrected - cx) * z / fx
    y = (y_px_corrected - cy) * z / fy
    return x, y, z


def test_coordinates(T, points_from_kinect, points_from_robot):
    print(T, '\n')
    for i in range(len(points_from_kinect) - 4):
        coord_from_kinect = points_from_kinect[4 + i]
        coord_from_kinect = [*coord_from_kinect, 1]
        coord = T @ np.transpose(coord_from_kinect)
        real_robot_coords = points_from_robot[4 + i]
        print(coord[:3].astype(int))
        print(*real_robot_coords)
        print('error', np.linalg.norm(coord[:3] - [*real_robot_coords]), '\n')


def calc_transform_matrix(points_from_kinect, points_from_robot):
    a, b, c, d = points_from_kinect[:4]
    x1, y1, z1 = a[:]
    x2, y2, z2 = b[:]
    x3, y3, z3 = c[:]
    x4, y4, z4 = d[:]

    A, B, C, D = points_from_robot[:4]
    X1, Y1, Z1 = A[:]
    X2, Y2, Z2 = B[:]
    X3, Y3, Z3 = C[:]
    X4, Y4, Z4 = D[:]

    transformMatrix = get_transform(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4,
                                    X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4)
    T = np.vstack((transformMatrix, [0, 0, 0, 1]))
    return T


def generate_sample_points():
    transform = np.array([
        [3.41949594e-01, -2.80219746e-01, -8.90750570e-01, 2.04391318],
        [-9.42365806e-01, -6.98786949e-02, -3.53852884e-01, 1.23556853],
        [-1.18430076e-02, 9.62697673e-01, -2.80952091e-01, 0.960580436],
        [0, 0, 0, 1]
    ])
    generator = PointGenerator()
    return generator.generate_robot_points(
        min_distance=0.8,
        max_distance=1.8,
        fov_degrees=(60, 58),
        distance_slice_count=10,
        point_counts_in_slice=(10, 10),
        transform_matrix=transform,
        max_radius=1.35,
        tool_vertical_offset=0.1,
        min_height=0.6,
        min_cylinder_radius=0.2)


class DepthRuntime(object):
    def __init__(self):
        pygame.init()

        pygame.display.set_caption("Kinect for Windows v2 Depth")

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self._done = False

        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect depth frames, 8bit grey,
        # width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface(
            (self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode(
            (self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

    def draw_depth_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        f8 = np.uint8(frame.clip(1, 4000) / 16.)
        frame8bit = np.dstack((f8, f8, f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def run(self):
        last_points_of_interest = np.zeros((50,3))
        last_frames = np.zeros((3, frame_width*frame_height))

        points_from_robot = []
        points_from_kinect = []

        dataset = []


        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self._done = True  # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE:  # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                # frame = np.flip(frame.reshape((frame_height, frame_width)), 1).reshape(-1)

                frame[:5120] = 0
                frame[-60000:] = 0
                frame[frame < 800] = 0
                frame[frame > 4500] = 0

                last_frames = enqueue(frame, last_frames)
                filtered_frame = noise_filter(last_frames)

                top_point_index = find_top_point_index(filtered_frame)
                # OR top_point_index = find_closest_element(filtered_frame)

                top_point = (top_point_index % frame_width, top_point_index // frame_width, filtered_frame[top_point_index])
                last_points_of_interest = enqueue(top_point, last_points_of_interest)
                mean_point = last_points_of_interest.mean(axis=0)
                # print(mean_x, mean_y, top_point_index)

                self.get_user_input(dataset, filtered_frame, mean_point, points_from_kinect, points_from_robot)

                self.draw_depth_frame(filtered_frame, self._frame_surface)

                pygame.draw.circle(self._frame_surface, (255, 0, 0), (int(mean_point[0]), int(mean_point[1])), 10, 1)
            self._screen.blit(self._frame_surface, (0, 0))
            pygame.display.update()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()

    def get_user_input(self, dataset, frame, mean_point, points_from_kinect, points_from_robot):
        mean_point_in_mm = calc_coords_in_mm(*mean_point)
        # print('my_coords:', kx, ky, kz)
        # print('Alex_coords:', ckx, cky, ckz)
        # print('Alex+calibration_coords:', cckx, ccky, cckz)

        if pygame.key.get_pressed()[pygame.K_q]:
            points_from_kinect.append(mean_point_in_mm)
            pos = robot.get_pos()
            print(mean_point_in_mm)
            print([*pos * 1000], '\n')
            points_from_robot.append(pos * 1000)
            sleep(0.2)

        if pygame.key.get_pressed()[pygame.K_r]:
            self.get_initial_transform_matrix(points_from_kinect, points_from_robot, mean_point_in_mm)

        if pygame.key.get_pressed()[pygame.K_e]:
            robot_points, kinect_points = generate_sample_points()

            def robot_routine(robot_points):
                robot_poses = np.pad(robot_points, ((0, 0), (0, 3)), 'constant')
                robot.movej([0, -1.57, 0, 0, 1.57, 0], acc=0.4, vel=0.4)
                for pose in robot_points:
                    robot.movex('movej', robot_poses, acc=0.4, vel=0.4)
                    sleep(5)

            threading.Thread(target=robot_routine, args=(robot_points)).start()
            sleep(0.2)

        if pygame.key.get_pressed()[pygame.K_w]:
            print(*points_from_kinect)
            print(*points_from_robot, '\n')
            sleep(0.1)

        if pygame.key.get_pressed()[pygame.K_t]:
            transform = calc_transform_matrix(points_from_kinect, points_from_robot)
            print("TransformMatrix")
            print(transform, '\n')

            test_coordinates(transform, points_from_kinect, points_from_robot)

        if pygame.key.get_pressed()[pygame.K_x]:
            f = frame.reshape((frame_height, frame_width))
            np.savetxt('kinect-%s.txt' % time(), f, fmt='% 4s')
        if pygame.key.get_pressed()[pygame.K_a]:
            f = frame.reshape((frame_height, frame_width))
            dataset.append(f)
        if pygame.key.get_pressed()[pygame.K_s]:
            np.save('kinect-dataset', dataset)
        if pygame.key.get_pressed()[pygame.K_d]:
            npz_dataset = np.load('kinect-dataset.npy')
            print(npz_dataset.shape)

    @staticmethod
    def get_initial_transform_matrix(points_from_kinect, points_from_robot, mean_point_in_mm):
        def robot_routine():
            poselist = ([0.0248, -0.1748, 1.3079, 0, 0, 0],
                        [0.4091, 0.8586, 0.8775, 0, 0, 0],
                        [-0.3577, 0.7804, 0.8975, 0, 0, 0],
                        [0.8673, -0.2957, 0.8791, 0, 0, 0],

                        [0.4372, 0.18387, 1.1031, 0, 0, 0],
                        [-0.5041, 0.5650, 0.9561, 0, 0, 0],
                        [-0.1466, 0.1610, 0.6727, 0, 0, 0])

            robot.movej([0, -1.57, 0, 0, 1.57, 0], acc=0.4, vel=0.4)
            for pose in poselist:
                robot.movex('movej', pose, acc=0.4, vel=0.4)
                sleep(5)
                points_from_kinect.append(mean_point_in_mm)
                pos = robot.get_pos()
                points_from_robot.append(pos * 1000)
                print(mean_point_in_mm)

                print([*pos * 1000], '\n')

        threading.Thread(target=robot_routine, ).start()
        sleep(0.2)


__main__ = "Kinect v2 Depth"
robot = urx.Robot("10.0.0.2", use_rt=True)
robot.set_tcp((0, 0, 0.1, 0, 0, 0))
game = DepthRuntime()
game.run()
