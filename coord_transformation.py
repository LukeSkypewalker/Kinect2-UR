import numpy as np
from sympy import *


# x1 - coordinates from kinect
# X1 - coordinates from Robot
#
# Test Case:
# TransformMatrix = someTransformation(8, 1, 1,  2, 2, 3,  3, 3, 2,  7, 7, 8,    1, 7, 1,  2, 1, 3,  3, 2, 2,  7, 6, 8)
# [[ 0.  1.  0.  0.]
# [ 1.  0.  0. -1.]
# [ 0.  0.  1.  0.]]


def get_transform(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4):
    r11 = Symbol('r11')
    r12 = Symbol('r12')
    r13 = Symbol('r13')
    r21 = Symbol('r21')
    r22 = Symbol('r22')
    r23 = Symbol('r23')
    r31 = Symbol('r31')
    r32 = Symbol('r32')
    r33 = Symbol('r33')
    t1 = Symbol('t1')
    t2 = Symbol('t2')
    t3 = Symbol('t3')

    eq1 = r11 * x1 + r12 * y1 + r13 * z1 - X1 + t1
    eq2 = r21 * x1 + r22 * y1 + r23 * z1 - Y1 + t2
    eq3 = r31 * x1 + r32 * y1 + r33 * z1 - Z1 + t3

    eq4 = r11 * x2 + r12 * y2 + r13 * z2 - X2 + t1
    eq5 = r21 * x2 + r22 * y2 + r23 * z2 - Y2 + t2
    eq6 = r31 * x2 + r32 * y2 + r33 * z2 - Z2 + t3

    eq7 = r11 * x3 + r12 * y3 + r13 * z3 - X3 + t1
    eq8 = r21 * x3 + r22 * y3 + r23 * z3 - Y3 + t2
    eq9 = r31 * x3 + r32 * y3 + r33 * z3 - Z3 + t3

    eq10 = r11 * x4 + r12 * y4 + r13 * z4 - X4 + t1
    eq11 = r21 * x4 + r22 * y4 + r23 * z4 - Y4 + t2
    eq12 = r31 * x4 + r32 * y4 + r33 * z4 - Z4 + t3

    res = (solve([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12],
                 [r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3]))
    print(res)
    r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3 = \
        res[r11], res[r12], res[r13], res[r21], res[r22], res[r23], res[r31], res[r32], res[r33], res[t1], res[t2], res[
            t3]
    tranform_matrix = np.array([[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3]], dtype='float')
    return tranform_matrix


if __name__ == '__main__':
    # TransformMatrix = get_transform(8, 1, 1,   2, 2, 3,   3, 3, 2,   7, 7, 8,      1, 7, 1,   2, 1, 3,   3, 2, 2,   7, 6, 8)
    # TransformMatrix = get_transform(1125, 1052, 2136,    1406, 671, 1624,    2554, 1317, 1970,    2957, 1174, 2565,
    #                                404, 758, 1109,      -451, 839, 1109,    -992, 15, 1002,      -544, -536, 1163)
    #
    TransformMatrix = get_transform(-804, 382, 2033,    -773, 217, 1128,    372, 362, 1250.71,    200, 754, 1689,
                                    -257, 780, 797,      609, 858, 876,    567, -295, 979,      24, -175, 1247)

    T = [[-9.36647127e-02, 1.90660064e-01, -9.58894013e-01, 1.46758372e+03],
         [9.85362837e-01, 3.44422261e-02, -8.00284368e-02, 6.15954505e+02],
         [-3.65105791e-02, -1.01721987e+00, -1.51485105e-01, 9.85841789e+02],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    coord_from_kinect = [-312.2315761076157, -182.90837796973037, 1429, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [97.42617785290247, 185.36119512326243, 946.41153016294095]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')

    coord_from_kinect = [-772.4821217533466, -250.20781123489084, 1225, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [312.93529139264638, -266.58537503013537, 1042.0696680745425]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')

    coord_from_kinect = [-161.6431977219928, -104.71282004877882, 1538, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-7.4884398059476203, 321.96199693690039, 855.41792929956023]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')



'''
[-804.12783510546683, 382.05347951926876, 2033.1600000000001]
[-0.78931711199159393, -0.34712528512324331, 2.0331600000000001]
[2190866735422.7864, 2772512771376.3906, 2.0331600000000001]
[-257.69777249900307, 780.57829305684368, 797.47115940353012] 

[-773.40828716037515, 217.52245092698902, 1128.47]
[-0.76170242836346913, -0.19816219257585047, 1.1284700000000001]
[19288022149.685856, 332177376595.68042, 1.1284700000000001]
[609.6190187074576, 858.79215355412589, 876.51780899566074] 

[372.24318450105113, 362.88818319189068, 1250.71]
[0.37211647837222545, -0.3420102662153584, 1.25071]
[348429044444970.06, 101056975955464.31, 1.25071]
[567.48777311274512, -295.7859813107907, 979.08409342785831] 

[200.94160077093284, 754.62512467423505, 1689.5999999999999]
[0.20398770631758686, -0.72767882316886301, 1.6896]
[101368535615431.08, 16273209730904.564, 1.6896]
[24.333977375581327, -175.04430734537681, 1247.6485970015942] 


[-162.21388804245893, 713.95339247334664, 2717.8099999999999]
[-0.15214131654699881, -0.66823042322468373, 2.7178100000000001]
[52818106313262.227, 26032778787791.902, 2.7178100000000001]
[-959.66914375391934, 92.361257543567191, 1022.0682834201847] 

[-0.15214131654699881, -0.66823042322468373, 2.7178100000000001]
[52818106313262.227, 26032778787791.902, 2.7178100000000001]
[-959.66914375391934, 92.361257543567191, 1022.0682834201847] 

[-98.770001080491483, 428.16980640608193, 1014.53]
[-0.094602779376997417, -0.41187104179436101, 1.0145299999999999]
[8197088642640.668, 2117209987725.1436, 1.0145299999999999]
[736.89683354883709, 183.543876621148, 1103.38853900971] 

[-98.770001080491483, 428.16980640608193, 1014.53]
[-0.094602779376997417, -0.41187104179436101, 1.0145299999999999]
[8197088642640.668, 2117209987725.1436, 1.0145299999999999]
[736.96942137464373, 183.59259896580622, 1103.3353765116506] 

[-596.29919528492144, 599.25779792052288, 2222.6700000000001]
[-0.58312123555229611, -0.56193794779672523, 2.2226699999999999]
[5065079860671.6289, 3599710844376.5449, 2.2226699999999999]
[-503.97722194665363, 565.08787811905245, 956.0115818046969] 

[-596.29919528492144, 599.25779792052288, 2222.6700000000001]
[-0.58312123555229611, -0.56193794779672523, 2.2226699999999999]
[5065079860671.6289, 3599710844376.5449, 2.2226699999999999]
[-503.97722194665363, 565.08787811905245, 956.0115818046969] 

[-156.506265182059, 234.13797456308251, 1953.96]
[-0.14883963727507549, -0.19993743213957363, 1.9539600000000001]
[58589910559120.617, 43354056439095.5, 1.9539600000000001]
[-156.506265182059, 234.13797456308251, 1953.96]




    TransformMatrix = [[-9.79827640e-02, 8.73191690e-02, - 9.89189247e-01, 1.49799232e+03],
                       [9.82666608e-01, - 2.65835318e-06, - 7.87607721e-01, 6.25481944e+02],
                       [-5.95851933e-03, - 6.60597146e-01, 4.28561556e-01, 9.74124466e+02]]

    print("TransformMatrix")
    print(TransformMatrix, '\n')
    T = np.vstack((TransformMatrix, [0, 0, 0, 1]))

    coord_from_kinect = [1020.02611935, 1092.33581591, 1528, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-43.1085493412, 423.586070608, 903.185964107]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')




    coord_from_kinect = [1018.02345027, 1090.19117752, 1525, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-43.1423689209, 423.588835174, 903.200642963]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')



    coord_from_kinect = [387.723354764, 790.844594675, 1326, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [212.703451173, - 33.9096297333, 1008.62247852]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')



    coord_from_kinect = [1744.10960805, 1665.12174995, 2316, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-797.348480053, 526.437169265, 888.382770722]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')



    coord_from_kinect = [1362.07979184, 979.425711374, 1296, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [155.230077374,944.834905473 ,873.972121575]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')



    TransformMatrix = [[4.19547303e-03, 2.16126013e-02, - 1.01134744e+00, 2.57638207e+03],
                       [9.87035102e-01, 3.27665863e-02, -7.20261977e-01, 2.71354406e+02],
                       [-5.78215782e-03, - 6.66352520e-01, 5.74266519e-01, 9.60981905e+02]]

    print("TransformMatrix")
    print(TransformMatrix, '\n')
    T = np.vstack((TransformMatrix, [0, 0, 0, 1]))

    coord_from_kinect = [1531.19774791, 1822.91811456, 2721, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-115.868416701, -117.101141445, 1294.88245336]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')

    coord_from_kinect = [2532.43298325, 2819.84407513, 3504, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-930.306532961, 316.077137844, 1066.55903638]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')

    coord_from_kinect = [2658.30321774, 2172.68616702, 2633, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [11.2656437289, 1042.84886031, 1006.12543066]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')

    coord_from_kinect = [439.949985258, 1349.10009305, 1611, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [983.29343919, -416.930753088, 983.144803852]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')




 TransformMatrix = [[-9.16918141e-01, -7.54609860e-03, 1.06167417e+00, -8.75896785e+02],
                       [-3.62030329e-01, 1.45992931e-02, -7.52184096e-01, 1.91113659e+03],
                       [4.95795816e-03, -3.59942484e-01, 4.10536110e-01, 1.04040392e+03]]
                       
                       
                    
    coord_from_kinect = [1184.60910576, 2122.33983661, 1760, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-100, 168, 995]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')

    coord_from_kinect = [2453.55099278, 3072.65728352, 2391, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-601, -728, 931]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')

    coord_from_kinect = [864.582034682, 1777.87330747, 1393, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    real_robot_coords = [-214, 589, 960]
    print(*real_robot_coords)
    print('error', np.linalg.norm(coord[:3] - real_robot_coords), '\n')



TransformMatrix = get_transform(1125, 1052, 2136,    1406, 671, 1624,    2554, 1317, 1970,    2957, 1174, 2565,
    #                                404, 758, 1109,      -451, 839, 1109,    -992, 15, 1002,      -544, -536, 1163)
    coord_from_kinect = [2760, 2116, 3005, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    print(98, -535, 970, '\n')

    coord_from_kinect = [2831, 2126, 3355, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    print(474, -741, 1045, '\n')

    coord_from_kinect = [1559, 1630, 1570, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    print(-684, 841, 812, '\n')

    coord_from_kinect = [1224, 1932, 1557, 1]
    coord = T @ np.transpose(coord_from_kinect)
    print(*coord[:3].astype(int))
    print(-449, 1046, 712, '\n')
'''
