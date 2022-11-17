import numpy as np
import transform as trans
import cv2

class Init():
    def __init__(self, unit_size, marker_size):
        self.marker_lenght = marker_size
        self.unit_size = unit_size
        self.createMarkerEdge()
        self.init_offset_matrix()
        self.createArucoParameters()

    def getMarkerEdge(self):
        return self.aruco_edges

    def getMatrixOffset(self):
        return self.offset_matrix

    def getMarkerSize(self):
        return self.marker_lenght

    def getArucoParameters(self):
        return self.aurocParameters

    def createArucoParameters(self):
        self.aurocParameters = cv2.aruco.DetectorParameters_create()
        # self.aurocParameters.adaptiveThreshWinSizeMin = 4
        # self.aurocParameters.adaptiveThreshWinSizeMax = 26
        # self.aurocParameters.adaptiveThreshWinSizeStep = 2
        # self.aurocParameters.minMarkerPerimeterRate  = 0.01
        # self.aurocParameters.maxMarkerPerimeterRate  = 4
        # self.aurocParameters.adaptiveThreshWinSizeStep = 2
        # self.aurocParameters.polygonalApproxAccuracyRate  = 0.1
        # self.aurocParameters.perspectiveRemovePixelPerCell  = 10
        # self.aurocParameters.cornerRefinementWinSize = 10
        # self.aurocParameters.cornerRefinementMethod = 1
        # self.aurocParameters.cornerRefinementMinAccuracy = 0.01
        self.aurocParameters.cornerRefinementMaxIterations = cv2.aruco.CORNER_REFINE_SUBPIX

    def createMarkerEdge(self):
        self.aruco_edges = np.array([[-self.marker_lenght / 2, self.marker_lenght / 2, 0],
                            [self.marker_lenght / 2, self.marker_lenght / 2, 0],
                            [self.marker_lenght / 2, -self.marker_lenght / 2, 0],
                            [-self.marker_lenght / 2, -self.marker_lenght / 2, 0]],
                           dtype='float32').reshape((4, 1, 3))

    def init_offset_matrix(self):
        rot_offset = np.array( [[0,   2.221449705, 2.221449705 ],
                                [-0.7853982, 0,    0   ],
                                [-0.6139431,  1.4821898,   0.6139431 ],
                                [0, 2.9024532,  1.2022355     ],
                                [-0.6139431,  -1.4821898,  -0.6139431  ],
                                [0,    0,    0   ],
                                [0,    0.785398163 , 0   ],
                                [0,    1.570796327 , 0   ],
                                [0,    2.35619449 , 0   ],
                                [0,    3.14159,  0   ],
                                [0,    -2.35619449, 0   ],
                                [0,    -1.570796327, 0   ],
                                [0,    -0.785398163, 0   ],
                                [0.7853982,  0,    0   ],
                                [0.6139431,  1.4821898,   -0.6139431  ],
                                [0,    2.9024532,   -1.2022355 ],
                                [0.6139431,  -1.4821898,  0.6139431 ],
                                [1.2091996,  1.2091996,   -1.2091996 ]])


        size_0_5 = self.unit_size * 0.5
        cos_45 = np.sin(45 * np.pi/180) * size_0_5

        pos_offset = np.array( [[0,         size_0_5,   0       ],
                                [0,         cos_45,     cos_45  ],
                                [cos_45,    cos_45,     0       ],
                                [0,         cos_45,     -cos_45 ],
                                [-cos_45,   cos_45,     0       ],
                                [0,         0,          size_0_5],
                                [cos_45,    0,          cos_45  ],
                                [size_0_5,  0,          0       ],
                                [cos_45,    0,          -cos_45  ],
                                [0,         0,          -size_0_5],
                                [-cos_45,   0,          -cos_45  ],
                                [-size_0_5, 0,          0       ],
                                [-cos_45,   0,          cos_45  ],
                                [0,         -cos_45,    cos_45  ],
                                [cos_45,    -cos_45,    0       ],
                                [0,         -cos_45,    -cos_45 ],
                                [-cos_45,   -cos_45,    0       ],
                                [0,         -size_0_5,  0       ]])
        self.offset_matrix = []
        for i in range(0,len(pos_offset)):
            self.offset_matrix.append(trans.rvecTvecToTransfMatrix(tvec=pos_offset[i],rvec=rot_offset[i]))

