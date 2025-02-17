"""Script to linearize symbolic dynamics of 3D Double Ball Balancer"""

import pickle

from symbolic_dynamics import R_IB2, ang, omega, omega_cmd, omega_dot
from sympy import Matrix, pi, symbols
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3

# select equilibrium beta_z
beta_z_eq = 0 * pi

beta_x, beta_y, beta_z = symbols("beta_x beta_y beta_z")
R_IB2_beta = rot_axis3(-beta_z) * rot_axis2(-beta_y) * rot_axis1(-beta_x)

dyn = pickle.load(open("dynamics.p", "rb"))

for i in range(R_IB2.rows):
    for j in range(R_IB2.cols):
        dyn = dyn.subs(R_IB2[i, j], R_IB2_beta[i, j])

ang = ang.row_insert(3, Matrix([beta_x, beta_y, beta_z]))

# linearize system around equilibrium [0, ... ,beta_z_eq, ... , 0]
eq = [(x, 0 if x != beta_z else beta_z_eq) for x in ang]
eq.extend([(x, 0) for x in omega])
eq.extend([(x, 0) for x in omega_dot])
eq.extend([(x, 0) for x in omega_cmd])

if __name__ == "__main__":
    dyn_lin = dyn.subs(eq)

    for el in eq:
        print("add term for {}".format(el[0]))
        dyn_lin += dyn.jacobian(Matrix([el[0]])).subs(eq) * el[0]

    pickle_file = "linear_dynamics.p"
    print("write dynamics to {}".format(pickle_file))
    pickle.dump(dyn_lin, open(pickle_file, "wb"))
