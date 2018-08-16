from time import sleep, time

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import ctypes
import pygame
import numpy as np
import urx
from coord_transformation import get_transform

frame_width = 512
frame_height = 424
frame_width_half = frame_width / 2
frame_height_half = frame_height / 2
fov_horiz = np.deg2rad(70.6)
fov_vert = np.deg2rad(60)
fov_horiz_half = fov_horiz / 2
fov_vert_half = fov_vert / 2
T = []


def find_closest_element_index(frame):
    valid_idx = np.where(frame > 0)[0]
    return valid_idx[frame[valid_idx].argmin()]


def find_upper_element_index(frame):
    unit_frame = frame.copy()
    unit_frame[unit_frame > 0] = 1
    return unit_frame.argmax()


def noise_filter(frame, prev1_frame, prev2_frame,prev3_frame):
    #frame[:10000] = 0
    frame[-60000:] = 0
    frame[frame < 800] = 0
    frame[frame > 3000] = 0
    frame_copy = frame.copy()

    if prev3_frame is not None:
        mask0 = frame > 0
        mask1 = prev1_frame > 0
        mask2 = prev2_frame > 0
        mask3 = prev3_frame > 0
        mask = mask0 & mask1 & mask2 & mask3
        frame[~mask] = 0

    prev3_frame = prev2_frame
    prev2_frame = prev1_frame
    prev1_frame = frame_copy
    return frame, prev1_frame, prev2_frame, prev3_frame


def calc_mean_coords(frame, X, Y, Z, min_ind):
    x = min_ind % 512
    X = np.roll(X, 1)
    X[0] = x
    mean_x = X.mean()

    y = min_ind // 512
    Y = np.roll(Y, 1)
    Y[0] = y
    mean_y = Y.mean()

    z = frame[min_ind]
    Z = np.roll(Z, 1)
    Z[0] = z
    mean_z = Z.mean()

    return X, Y, Z, mean_x, mean_y, mean_z


def calc_calibrated_coords_in_mm(x_in_pixels, y_in_pixels, depth):
    cx = 254.878;
    cy = 205.395;
    fx = 365.456;
    fy = 365.456;
    k1 = 0.0905474;
    k2 = -0.26819;
    k3 = 0.0950862;
    p1 = 0.0;
    p2 = 0.0;

    z = depth/1000
    x = (x_in_pixels - cx) * z / fx;
    y = (y_in_pixels - cy) * z / fy;
    return x, y, z


def calc_coords_in_mm(x_in_pixels, y_in_pixels, depth):
    x_half = x_in_pixels - frame_width_half
    y_half = y_in_pixels - frame_height_half
    x_in_mm = depth * np.math.tan(fov_horiz_half) * (x_half / frame_width_half)
    y_in_mm = depth * np.math.tan(fov_vert_half) * (y_half / frame_height_half)
    return x_in_mm, y_in_mm, depth


def calc_coords_in_mm(x_in_pixels, y_in_pixels, depth):
    x_half = x_in_pixels - frame_width_half
    y_half = y_in_pixels - frame_height_half
    x_in_mm = depth * np.math.tan(fov_horiz_half) * (x_half / frame_width_half)
    y_in_mm = depth * np.math.tan(fov_vert_half) * (y_half / frame_height_half)
    return x_in_mm, y_in_mm, depth


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
        X = np.zeros(100)
        Y = np.zeros(100)
        Z = np.zeros(100)

        prev1_frame = None
        prev2_frame = None
        prev3_frame = None

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
                frame, prev1_frame, prev2_frame, prev3_frame = noise_filter(frame, prev1_frame, prev2_frame, prev3_frame)

                min_ind = find_upper_element_index(frame)
                # OR min_ind = find_closest_element(frame)

                X, Y, Z, mean_x, mean_y, mean_z = calc_mean_coords(frame, X, Y, Z, min_ind)
                # print(mean_x, mean_y, min_ind)

                kx, ky, kz = calc_coords_in_mm(mean_x, mean_y, mean_z)

                ckx, cky, ckz = calc_calibrated_coords_in_mm(mean_x, mean_y, mean_z)

                print('my_coords:', kx, ky, kz)
                print('Alex_coords:', ckx, cky, ckz)

                # check_user_input an print data
                if pygame.key.get_pressed()[pygame.K_q]:
                    points_from_kinect.append((kx, ky, kz))
                    #pos = robot.get_pos()
                    print([kx,ky,kz])
                    #print([*pos*1000], '\n')
                    #points_from_robot.append(pos * 1000)
                    sleep(0.2)

                if pygame.key.get_pressed()[pygame.K_w]:
                    print(*points_from_kinect)
                    print(*points_from_robot, '\n')
                    sleep(0.1)

                if pygame.key.get_pressed()[pygame.K_r]:
                    coord_from_kinect = [*points_from_kinect[-1], 1]
                    coord = T @ np.transpose(coord_from_kinect)
                    real_robot_coords = points_from_robot[-1]
                    print(coord[:3].astype(int))
                    print(*real_robot_coords)
                    print('error', np.linalg.norm(coord[:3] - [*real_robot_coords]), '\n')
                    pose = [*coord[:3]/1000, 0, 0, 0]
                    print(pose)
                    #robot.movel(pose, acc=0.2, vel=0.2)

                if pygame.key.get_pressed()[pygame.K_t]:
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

                    transformMatrix = get_transform(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4)
                    T = np.vstack((transformMatrix, [0, 0, 0, 1]))
                    print("TransformMatrix")
                    print(T, '\n')

                    for i in range(len(points_from_kinect) - 4):
                        coord_from_kinect = points_from_kinect[4+i]
                        coord_from_kinect = [*coord_from_kinect,1]
                        coord = T @ np.transpose(coord_from_kinect)
                        real_robot_coords = points_from_robot[4+i]
                        print(coord[:3].astype(int))
                        print(*real_robot_coords)
                        print('error', np.linalg.norm(coord[:3]-[*real_robot_coords]), '\n')


                if pygame.key.get_pressed()[pygame.K_x]:
                    f = frame.reshape((frame_height, frame_width))
                    np.savetxt('kinect-%s.txt' %time() , f, fmt='% 4s')

                if pygame.key.get_pressed()[pygame.K_a]:
                    f = frame.reshape((frame_height, frame_width))
                    dataset.append(f)

                if pygame.key.get_pressed()[pygame.K_s]:
                    np.save('kinect-dataset', dataset)

                if pygame.key.get_pressed()[pygame.K_d]:
                    npz_dataset = np.load('kinect-dataset.npy')
                    print(npz_dataset.shape)



                self.draw_depth_frame(frame, self._frame_surface)

                pygame.draw.circle(self._frame_surface, (255, 0, 0), (int(mean_x), int(mean_y)), 10, 1)
            self._screen.blit(self._frame_surface, (0, 0))
            pygame.display.update()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Depth"
#robot = urx.Robot("10.0.0.2", use_rt=True)
#robot.set_tcp((0, 0, 0.1, 0, 0, 0))
game = DepthRuntime();
game.run();
