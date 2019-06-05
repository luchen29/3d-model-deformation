import numpy as np
import math
def eulerAnglesToRotationMatrix(theta) :
    # rotate about x axis by angle theta[0]
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                  
    # rotate about y axis by angle theta[1]                 
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    
    # rotate about z axis by angle theta[2]            
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    # rotation about any arbitary axis can be written using matrix multiplication   
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


# euler=[90,0,0]
# print('===============')
# print('Rotation Matrix of euler=[90,0,0]: ')
# print(eulerAnglesToRotationMatrix(euler))
# print('===============')
# print('Pi: ')
# pi= math.pi
# print(pi)