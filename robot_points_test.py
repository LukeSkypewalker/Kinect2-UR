import time
import kinect_point_generator as kpg
import urx
import numpy as np

robot = urx.Robot("10.0.0.2", use_rt=True)
robot.set_tcp((0, 0, 0.1, 0, 0, 0))

transform = np.array([
    [3.41949594e-01, -2.80219746e-01, -8.90750570e-01, 2.04391318],
    [-9.42365806e-01, -6.98786949e-02, -3.53852884e-01, 1.23556853],
    [-1.18430076e-02, 9.62697673e-01, -2.80952091e-01, 0.960580436],
    [0, 0, 0, 1]
])

generator = kpg.PointGenerator()
robot_points, _ = generator.generate_robot_points(
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

# print(robot_points.shape)
# robot.movexs('movej', points, acc=0.2, vel=0.1)
robot_poses = np.pad(robot_points, ((0, 0), (0, 3)), 'constant')
# np.set_printoptions(threshold=np.nan)
# print('Reachable points:\n', np.array2string(result, separator=',', max_line_width=100))
robot.movej([0, -2, -1, -1.5, -1.5, -2], acc=0.3, vel=0.3)
for i, p in enumerate(robot_poses):
    print(time.strftime('%H:%M:%S',time.localtime()), i, p)
    robot.movex('movej', p, acc=0.4, vel=0.4)
    time.sleep(1)
