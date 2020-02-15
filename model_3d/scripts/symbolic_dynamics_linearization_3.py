"""Script to linearize symbolic dynamics of 3D Double Ball Balancer"""

import pickle
from sympy import Matrix, BlockMatrix, block_collapse, rot_axis3, rot_axis2, solve, pi

from symbolic_dynamics_3 import ang, ang_dot, omega, omega_dot, omega_cmd, beta_z, beta_y, beta_x_dot, beta_y_dot, beta_z_dot, R_IB3, B3_omega_3, w_2y, phi_y_dot, omega_y_cmd

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
beta_z_eq = 0.0 * pi
w2_y_eq = 0.0 * pi
phi_y_dot_eq = -w2_y_eq
omega_y_cmd_eq = -w2_y_eq

# linearize system around equilibrium [0, ... ,beta_z_eq, ... , 0]
eq = [(x, 0) if x != beta_z else (x, beta_z_eq) for x in ang]
eq.extend([(x, w2_y_eq) if x == w_2y else (x, phi_y_dot_eq)
           if x == phi_y_dot else (x, 0) for x in omega])
eq.extend([(x, 0) for x in omega_dot])
eq.extend([(x, 0) if x != omega_y_cmd else (x, omega_y_cmd_eq) for x in omega_cmd])

# compose linearized equations
dyn_lin = dyn.subs(eq)

for el in eq:
    print('add term for {}'.format(el[0]))
    dyn_lin += dyn.jacobian(Matrix([el[0]])).subs(eq) * el[0]

pickle_file = 'linear_dynamics.p'
print('write dynamics to {}'.format(pickle_file))
pickle.dump(dyn_lin, open(pickle_file, "wb"))
dyn_lin = pickle.load(open("linear_dynamics.p", "rb"))

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
