import threading
from time import sleep, time

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import ctypes
import pygame
import numpy as np
import urx
from coord_transformation import calc_transform_matrix
from kinect_point_generator import PointGenerator

FRAME_WIDTH = 512
FRAME_HEIGHT = 424
FOV_HORIZ = np.deg2rad(70.6)
FOV_VERT = np.deg2rad(60)
ERROR_RADIUS_M = 0.2
mean_point = None

transform_matrix = np.array([
    [3.18793477e-01, -2.11661971e-01, -9.67004001e-01, 2.16169931],
    [-9.44977143e-01, -7.73691372e-02, -3.43067985e-01, 1.21063893],
    [-9.53204265e-03, 9.61256842e-01, -2.74046180e-01, 0.961987136],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])


def find_closest_element_index(frame):
    valid_idx = np.where(frame > 0)[0]
    return valid_idx[frame[valid_idx].argmin()]


def find_top_point_index(frame):
    unit_frame = frame.copy()
    unit_frame[unit_frame > 0] = 1
    return unit_frame.argmax()


def noise_filter(last_frames):
    mask = np.all(last_frames > 0, 0)
    filtered_frame = np.copy(last_frames[0])
    filtered_frame[~mask] = 0
    return filtered_frame


def enqueue(item, array):
    result = np.roll(array, 1, 0)
    result[0] = item
    return result


def calc_coords_in_meters(x_in_pixels, y_in_pixels, depth_in_meters):
    x_half = x_in_pixels - FRAME_WIDTH / 2
    y_half = -(y_in_pixels - FRAME_HEIGHT / 2)
    x_in_meters = depth_in_meters * np.math.tan(FOV_HORIZ / 2) * (x_half / FRAME_WIDTH / 2)
    y_in_meters = depth_in_meters * np.math.tan(FOV_VERT / 2) * (y_half / FRAME_HEIGHT / 2)
    return x_in_meters, y_in_meters, depth_in_meters


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


def get_user_input(dataset, original_frame, filtered_frame, points_from_kinect, points_from_robot, robot_thread_data):
    global mean_point
    if pygame.key.get_pressed()[pygame.K_q]:
        mean_point_in_m = calc_coords_in_meters(*mean_point)
        points_from_kinect.append(mean_point_in_m)
        robot_position_m = robot.get_pos()
        print(mean_point_in_m)
        print([*robot_position_m], '\n')
        points_from_robot.append(robot_position_m)
        sleep(0.2)

    if pygame.key.get_pressed()[pygame.K_r]:
        get_poins_for_initial_transform_matrix(points_from_kinect, points_from_robot)

    if pygame.key.get_pressed()[pygame.K_e]:
        get_many_points(robot_thread_data)

    if pygame.key.get_pressed()[pygame.K_w]:
        print(*points_from_kinect)
        print(*points_from_robot, '\n')
        sleep(0.1)

    if pygame.key.get_pressed()[pygame.K_t]:
        transform_matrix = calc_transform_matrix(points_from_kinect, points_from_robot)
        print("TransformMatrix")
        print(transform_matrix, '\n')

        test_coordinates(transform_matrix, points_from_kinect, points_from_robot)

    if pygame.key.get_pressed()[pygame.K_x]:
        f = filtered_frame.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        np.savetxt('kinect-%s.txt' % time(), f, fmt='% 4s')

    if pygame.key.get_pressed()[pygame.K_a]:
        f = filtered_frame.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        dataset.append(f)

    if pygame.key.get_pressed()[pygame.K_g]:
        dataset.append((original_frame, filtered_frame))

    if pygame.key.get_pressed()[pygame.K_s]:
        np.save('kinect-dataset', dataset)

    if pygame.key.get_pressed()[pygame.K_d]:
        npz_dataset = np.load('kinect-dataset.npy')
        print(npz_dataset.shape)


def get_many_points(robot_thread_data):
    robot_points, kinect_points = generate_sample_points(transform_matrix)
    robot_thread_data['kinect_points'] = kinect_points

    points_from_kinect = []
    points_from_robot = []

    def robot_routine(robot_points, robot_thread_data):
        global mean_point
        robot_poses = np.pad(robot_points, ((0, 0), (0, 3)), 'constant')
        robot.movej([0, -2.2, -0.8, -2, -2, 0], acc=0.4, vel=0.4)
        for i, pose in enumerate(robot_poses):
            robot_thread_data['target_pose_index'] = i
            robot.movex('movej', pose, acc=0.4, vel=0.4)
            sleep(3)
            points_from_kinect.append(mean_point)
            pos = robot.get_pos()
            points_from_robot.append(pos)
            print(mean_point)
            print([*pos], '\n')
        # kinect observed (x_px, y_px, depth_m), robot from sensor (x_m, y_m, z_m)
        np.save('kinect-dataset', (points_from_kinect, points_from_robot))
        robot_thread_data['target_pose_index'] = None
        robot_thread_data['kinect_points'] = None

    threading.Thread(target=robot_routine, args=(robot_points, robot_thread_data)).start()
    sleep(0.3)


def get_poins_for_initial_transform_matrix(points_from_kinect, points_from_robot):
    points_from_kinect.clear()
    points_from_robot.clear()

    def robot_routine():
        global mean_point
        poselist = ([0.0248, -0.1748, 1.4079, 0, 0, 0],
                    [0.4091,  0.8586, 0.9775, 0, 0, 0],
                    [0.03577, 0.7804, 0.9975, 0, 0, 0],
                    [0.8673, -0.2957, 0.9791, 0, 0, 0],

                    [0.4372, 0.18387, 1.2031, 0, 0, 0],
                    [-0.3041, 0.5650, 1.0561, 0, 0, 0],
                    [-0.1466, 0.1610, 0.8727, 0, 0, 0])

        robot.movej([0, -1.57, 0, 0, 1.57, 0], acc=0.4, vel=0.4)
        for pose in poselist:
            robot.movex('movej', pose, acc=0.4, vel=0.4)
            sleep(5)
            mean_point_in_m = calc_coords_in_meters(*mean_point)
            points_from_kinect.append(mean_point_in_m)
            pos = robot.get_pos()
            points_from_robot.append(pos)
            print(mean_point_in_m)
            print([*pos], '\n')

    threading.Thread(target=robot_routine).start()
    sleep(0.2)


def generate_sample_points(transform):
    generator = PointGenerator()
    return generator.generate_robot_points(
        min_distance=0.8,
        max_distance=1.8,
        fov_degrees=(60, 58),
        distance_slice_count=10,
        point_counts_in_slice=(10, 10),
        transform_matrix=transform,
        max_radius=1.35,
        tool_vertical_offset=0.24,
        min_height=0.9,
        min_cylinder_radius=0.2)


class DepthRuntime(object):
    MILLIMETERS_IN_METER = 1000.0
    MAX_DISTANCE_TO_DRAW_M = 4.0
    MIN_DISTANCE_TO_SEARCH_M = 0.8
    MAX_DISTANCE_TO_SEARCH_M = 3.5
    INTENSITY_FACTOR = 255.0 / MAX_DISTANCE_TO_DRAW_M
    BLACK_BAR_TOP_HEIGHT = 10
    BLACK_BAR_BOTTOM_HEIGHT = 118

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

    @staticmethod
    def meters_to_pixels(length_in_meters, depth_in_meters, frame_size_in_pixels, fov_in_radians):
        # Assumptions: metric coordinates start at image center, with x going to the left and y to the top.
        frame_size_in_meters = 2 * depth_in_meters * np.tan(fov_in_radians / 2)
        raw_result = -frame_size_in_pixels / frame_size_in_meters * length_in_meters + frame_size_in_pixels / 2
        raw_result = np.clip(raw_result, 0, frame_size_in_pixels - 1)
        return int(round(raw_result))

    def draw_depth_frame(self, frame_in_meters, target_surface):
        if frame_in_meters is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        # Make farthest pixels the whitest.
        f8 = np.uint8(frame_in_meters.clip(0, self.MAX_DISTANCE_TO_DRAW_M) * self.INTENSITY_FACTOR)
        frame8bit = np.dstack((f8, f8, f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def run(self):
        global mean_point
        last_points_of_interest = np.zeros((50, 3))
        last_frames = np.zeros((3, FRAME_WIDTH * FRAME_HEIGHT))

        points_from_robot = []
        points_from_kinect = []

        dataset = []

        robot_thread_data = {'robot_points': [], 'kinect_points': None, 'target_pose_index': None}

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
                frame = self._kinect.get_last_depth_frame() / self.MILLIMETERS_IN_METER
                # frame = np.flip(frame.reshape((frame_height, frame_width)), 1).reshape(-1)

                point_of_interest_index = robot_thread_data['target_pose_index']

                trim_top, trim_bottom = (self.BLACK_BAR_TOP_HEIGHT, FRAME_HEIGHT - self.BLACK_BAR_BOTTOM_HEIGHT)
                trim_closest, trim_farthest = (self.MIN_DISTANCE_TO_SEARCH_M, self.MAX_DISTANCE_TO_SEARCH_M)
                if point_of_interest_index is not None:
                    point_of_interest = robot_thread_data['kinect_points'][point_of_interest_index]
                    depth_in_m = point_of_interest[2]
                    trim_top = DepthRuntime.meters_to_pixels(point_of_interest[1] + ERROR_RADIUS_M, depth_in_m, FRAME_HEIGHT, FOV_VERT)
                    trim_bottom = DepthRuntime.meters_to_pixels(point_of_interest[1] - ERROR_RADIUS_M, depth_in_m, FRAME_HEIGHT, FOV_VERT)
                    trim_closest = depth_in_m - ERROR_RADIUS_M
                    trim_farthest = depth_in_m + ERROR_RADIUS_M

                trimmed_frame = frame.copy()
                trimmed_frame[:FRAME_WIDTH * trim_top] = 0
                trimmed_frame[FRAME_WIDTH * trim_bottom:] = 0
                trimmed_frame[trimmed_frame < trim_closest] = 0
                trimmed_frame[trimmed_frame > trim_farthest] = 0

                last_frames = enqueue(trimmed_frame, last_frames)
                filtered_frame = noise_filter(last_frames)

                top_point_index = find_top_point_index(filtered_frame)
                top_point = (top_point_index % FRAME_WIDTH, top_point_index // FRAME_WIDTH, filtered_frame[top_point_index])
                last_points_of_interest = enqueue(top_point, last_points_of_interest)
                mean_point = last_points_of_interest.mean(axis=0)

                get_user_input(dataset, frame, filtered_frame, points_from_kinect, points_from_robot, robot_thread_data)

                self.draw_depth_frame(filtered_frame, self._frame_surface)

                pygame.draw.circle(self._frame_surface, (255, 0, 0), (int(mean_point[0]), int(mean_point[1])), 10, 1)
                pygame.draw.line(self._frame_surface, (255, 255, 0), (0, trim_top), (FRAME_WIDTH-1, trim_top), 2)
                pygame.draw.line(self._frame_surface, (255, 255, 0), (0, trim_bottom), (FRAME_WIDTH-1, trim_bottom), 2)
            self._screen.blit(self._frame_surface, (0, 0))
            pygame.display.update()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Depth"
robot = urx.Robot("10.0.0.2", use_rt=True)
robot.set_tcp((0, 0, 0.24, 0, 0, 0))
game = DepthRuntime()
game.run()
