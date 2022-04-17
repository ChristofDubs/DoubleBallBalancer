"""Parameters for 2D version of N-Ball Balancer

author: Christof Dubs
"""


def getDefaultParam(N: int):
    """Physical parameters of 2D N-Ball Balancer

    The N-Ball Balancer consists of N bodies:
        - N balls (subscript from 0 to N-1, with 0 being the ball on the bottom)
        - lever arm (inside ball with subscript N-1)

    Physical parameters that multiple bodies have are indexed accordingly.

    Attributes:
        N: total number of balls

    Returns: dictionary of parameters values (key: parameter name; value: parameter value)

        g: Gravitational constant [m/s^2]
        m_l: Mass of lever arm [kg]
        r_l: Arm length of lever [m] (distance from rotation axis to center of mass)
        theta_l: Mass moment of inertia of lever arm wrt. its center of mass [kg*m^2]

        m_0, m_1, ... , m_{N-1}: Mass of each ball [kg]
        r_0, r_1, ... , r_{N-1}: Radius of each ball [m]
        theta_0, theta_1, ... , theta_{N-1}: Mass moment of inertia of each ball wrt. its center of mass [kg*m^2]

        tau: time constant of speed controlled motor [s]
    """
    d = {"g": 9.81, "tau": 0.1, "r_l": 1.0, "m_l": 1.0, "theta_l": 1.0}
    for i in range(N):
        for prefix in ["r", "m", "theta"]:
            d[f'{prefix}_{i}'] = 1.0

    return d
