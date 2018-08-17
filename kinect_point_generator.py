import numpy as np

class PointGenerator:
    def _to_robot_point(self, kinect_point, transform_matrix):
        adjusted_point = np.append(kinect_point, 1)
        result = np.matmul(transform_matrix, adjusted_point)
        return result[:-1]

    def _generate_kinect_points(self,
                                min_distance=1.0,
                                max_distance=2.0,
                                fov_degrees=(70.6, 60),
                                distance_slice_count=3,
                                point_counts_in_slice=(3, 3)):
        result = []
        tangents = np.tan(np.deg2rad(fov_degrees) / 2)
        for z in np.linspace(min_distance, max_distance, distance_slice_count):
            max_offsets_in_slice = z * tangents
            # Create points in boustrophedon rows, top to bottom.
            x_direction = 1
            for y in np.linspace(max_offsets_in_slice[1], -max_offsets_in_slice[1], point_counts_in_slice[1]):
                start = x_direction * max_offsets_in_slice[0]
                for x in np.linspace(start, -start, point_counts_in_slice[0]):
                    new_point = np.array([x, y, z])
                    result.append(new_point)
                # Reverse the direction for the next row.
                x_direction = -x_direction
        return np.array(result)

    def _find_reachable_points(self,
                               src_points,
                               max_radius=1.35,
                               tool_vertical_offset=0.0,
                               min_height=0,
                               min_cylinder_radius=0.2):
        result = []
        for point in src_points:
            normalized_point = [point[0], point[1], point[2]- tool_vertical_offset]
            if np.linalg.norm(normalized_point) <= max_radius and \
                    point[2]> min_height and \
                    np.linalg.norm(point[:2]) > min_cylinder_radius:
                result.append(point)
        return np.array(result)

    def _to_robot_points(self,
                         kinect_points,
                         transform_matrix=np.identity(4)):
        result = []
        for kinect_point in kinect_points:
            robot_point = self._to_robot_point(kinect_point, transform_matrix)
            result.append(robot_point)
        return np.array(result)

    def generate_robot_points(self,
                              # point generation params
                              min_distance,
                              max_distance,
                              fov_degrees,
                              distance_slice_count,
                              point_counts_in_slice,
                              transform_matrix,

                              # point filtering params
                              max_radius,
                              tool_vertical_offset,
                              min_height,
                              min_cylinder_radius):

        kinect_points = self._generate_kinect_points(min_distance, max_distance, fov_degrees, distance_slice_count,
                                                     point_counts_in_slice)

        robot_points = self._to_robot_points(kinect_points, transform_matrix)
        reachable_points = self._find_reachable_points(robot_points, max_radius, tool_vertical_offset, min_height,
                                                       min_cylinder_radius)

        # Pad each point with default orientation orientation, e.g. [3,2,6] => [3,2,6,0,0,0]
        result = np.pad(reachable_points, ((0, 0), (0, 3)), 'constant')
        # np.set_printoptions(threshold=np.nan)
        # print('Reachable points:\n', np.array2string(result, separator=',', max_line_width=100))
        return result


transform = np.array([
    [9.11553612e-02, -2.97533421e-01, -9.00868653e-01, 1.76319118e+00],
    [-9.99589709e-01, 2.31756532e-02, -1.23105851e-01, 2.17456855e-01],
    [-8.01612547e-03, 9.80898800e-01, -2.65743949e-01, 9.56882597e-01],
    [0, 0, 0, 1]
])

generator = PointGenerator()
final_points = generator.generate_robot_points(
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

pass


