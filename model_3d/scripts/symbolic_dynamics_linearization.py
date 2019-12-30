"""Script to linearize symbolic dynamics of 3D Double Ball Balancer"""

import pickle
from sympy import symbols, Matrix

from symbolic_dynamics import ang, omega, omega_dot, omega_cmd

dyn = pickle.load(open("dynamics.p", "rb"))

# linearize system around equilibrium [0, ... , 0]
eq = [(x, 0) for x in ang]
eq.extend([(x, 0) for x in omega])
eq.extend([(x, 0) for x in omega_dot])
eq.extend([(x, 0) for x in omega_cmd])

dyn_lin = dyn.subs(eq)

for el in eq:
    print('add term for {}'.format(el[0]))
    dyn_lin += dyn.jacobian(Matrix([el[0]])).subs(eq) * el[0]

pickle_file = 'linear_dynamics.p'
print('write dynamics to {}'.format(pickle_file))
pickle.dump(dyn_lin, open(pickle_file, "wb"))
