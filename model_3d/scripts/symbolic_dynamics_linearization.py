"""Script to linearize symbolic dynamics of 3D Double Ball Balancer"""

import pickle
from sympy import symbols, BlockMatrix, Matrix, pi
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3

from symbolic_dynamics import ang, ang_dot, omega, omega_dot, omega_cmd, R_IB2, B2_omega_2

# load dynamics
dyn = pickle.load(open("dynamics.p", "rb"))

# substitute parameters with numerical values
dyn = dyn.subs("a", 0.1)
dyn = dyn.subs("l", 1.0)
dyn = dyn.subs("m1", 1.0)
dyn = dyn.subs("m2", 1.0)
dyn = dyn.subs("m3", 1.0)
dyn = dyn.subs("mu1", 0.8)
dyn = dyn.subs("mu12", 0.8)
dyn = dyn.subs("r1", 3.0)
dyn = dyn.subs("r2", 2.0)
dyn = dyn.subs("theta1", 1.0)
dyn = dyn.subs("theta2", 1.0)
dyn = dyn.subs("theta3x", 1.0)
dyn = dyn.subs("theta3y", 1.0)
dyn = dyn.subs("theta3z", 1.0)
dyn = dyn.subs("g", 9.81)
dyn = dyn.subs("tau", 0.1)

# select equilibrium beta_z
beta_z_eq = 0 * pi

beta_x, beta_y, beta_z = symbols('beta_x beta_y beta_z')
R_IB2_beta = rot_axis3(-beta_z) * rot_axis2(-beta_y) * rot_axis1(-beta_x)

for i in range(R_IB2.rows):
    for j in range(R_IB2.cols):
        dyn = dyn.subs(R_IB2[i, j], R_IB2_beta[i, j])

ang = ang.row_insert(3, Matrix([beta_x, beta_y, beta_z]))
ang_dot = ang_dot.row_insert(3, B2_omega_2)

# linearize system around equilibrium [0, ... ,beta_z_eq, ... , 0]
eq = [(x, 0 if x != beta_z else beta_z_eq) for x in ang]
eq.extend([(x, 0) for x in omega])
eq.extend([(x, 0) for x in omega_dot])
eq.extend([(x, 0) for x in omega_cmd])

dyn_lin = dyn.subs(eq)

for el in eq:
    print('add term for {}'.format(el[0]))
    dyn_lin += dyn.jacobian(Matrix([el[0]])).subs(eq) * el[0]

# solve A * omega_dot = b
A = dyn_lin.jacobian(omega_dot)
b = -dyn_lin.subs([(x, 0) for x in omega_dot])
omega_dot_lin = A.LUsolve(b)

# compose derivative vector of angular stats
ang_dot_lin = ang_dot.subs(eq)
for el in eq:
    ang_dot_lin += ang_dot.jacobian(Matrix([el[0]])).subs(eq) * el[0]

# state: [ang, omega]
state = Matrix(BlockMatrix([[ang], [omega]]))
state_derivative = Matrix(BlockMatrix([[ang_dot_lin], [omega_dot_lin]]))

A = state_derivative.jacobian(state)
B = state_derivative.jacobian(omega_cmd)

pickle_file = 'system_matrices.p'
print('write system matrices to {}'.format(pickle_file))
pickle.dump([A, B], open(pickle_file, "wb"))
