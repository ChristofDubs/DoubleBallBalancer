"""Script to linearize symbolic dynamics of 3D Double Ball Balancer"""

import pickle
from sympy import symbols, Matrix, pi

from symbolic_dynamics import ang, omega, omega_dot, omega_cmd, beta_z

# select equilibrium beta_z
beta_z_eq = 0 * pi

dyn = pickle.load(open("dynamics.p", "rb"))

# linearize system around equilibrium [0, ... ,beta_z_eq, ... , 0]
eq = [(x, 0 if x != beta_z else beta_z_eq) for x in ang]
eq.extend([(x, 0) for x in omega])
eq.extend([(x, 0) for x in omega_dot])
eq.extend([(x, 0) for x in omega_cmd])

if __name__ == '__main__':
    dyn_lin = dyn.subs(eq)

    for el in eq:
        print('add term for {}'.format(el[0]))
        dyn_lin += dyn.jacobian(Matrix([el[0]])).subs(eq) * el[0]

    pickle_file = 'linear_dynamics.p'
    print('write dynamics to {}'.format(pickle_file))
    pickle.dump(dyn_lin, open(pickle_file, "wb"))