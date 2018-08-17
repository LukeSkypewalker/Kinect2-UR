import numpy as np


class PointGenerator:
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
        robot_reachable_points, kinect_reachable_points = self._find_reachable_points(robot_points,
                                                                                      kinect_points,
                                                                                      max_radius,
                                                                                      tool_vertical_offset,
                                                                                      min_height,
                                                                                      min_cylinder_radius)

        # Pad each point with default orientation orientation, e.g. [3,2,6] => [3,2,6,0,0,0]

        return robot_reachable_points, kinect_reachable_points

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

    def _to_robot_points(self,
                         kinect_points,
                         transform_matrix=np.identity(4)):
        result = []
        for kinect_point in kinect_points:
            robot_point = self._to_robot_point(kinect_point, transform_matrix)
            result.append(robot_point)
        return np.array(result)

    def _to_robot_point(self, kinect_point, transform_matrix):
        adjusted_point = np.append(kinect_point, 1)
        result = np.matmul(transform_matrix, adjusted_point)
        return result[:-1]

    def _find_reachable_points(self,
                               robot_points,
                               kinect_points,
                               max_radius=1.35,
                               tool_vertical_offset=0.0,
                               min_height=0,
                               min_cylinder_radius=0.2):
        robot_reachable_points = []
        kinect_reachable_points = []
        for i, point in enumerate(robot_points):
            # todo point-tcp
            normalized_point = [point[0], point[1], point[2] - tool_vertical_offset]
            if np.linalg.norm(normalized_point) <= max_radius and \
                    point[2] > min_height and \
                    np.linalg.norm(point[:2]) > min_cylinder_radius:
                robot_reachable_points.append(point)
                kinect_reachable_points.append(kinect_points[i])
        return np.array(robot_reachable_points), np.array(kinect_reachable_points)
