"""Script to generate symbolic dynamics of 2D Double Ball Balancer

Derivation of the rigid multi-body dynamics using the Projected Newton-Euler method.
"""
from sympy import *

# angles
alpha, beta, phi, psi = symbols('alpha beta phi psi')
ang = Matrix([beta, phi, psi])

# angular velocities
alpha_dot, beta_dot, phi_dot, psi_dot = symbols('alpha_d beta_dot phi_dot psi_dot')
omega = Matrix([beta_dot, phi_dot, psi_dot])

# angular accelerations
beta_ddot, phi_ddot, psi_ddot = symbols('beta_dd phi_dd psi_dd')
omega_dot = Matrix([beta_ddot, phi_ddot, psi_ddot])

# parameter
l, m1, m2, m3, r1, r2, tau, theta1, theta2, theta3 = symbols(
    'l, m1, m2, m3, r1, r2, tau, theta1, theta2, theta3')

# constants
g = symbols('g')

# inputs
omega_cmd, T = symbols('omega_cmd T')

# parameter lists:
m = [m1, m2, m3]
theta = [theta1, theta2, theta3]

# kinematic constraints: lower ball rolling on the ground
x = -r1 * alpha
x_dot = -r1 * alpha_dot

# kinematic constraints: upper ball rolling on lower ball

om1 = Matrix([0, 0, alpha_dot])
r_S1P1 = Matrix([-r1 * sin(psi), r1 * cos(psi), 0])
v_OS1 = Matrix([-r1 * alpha_dot, 0, 0])

v_P1 = v_OS1 + om1.cross(r_S1P1)

r_OS2 = Matrix([x - sin(psi) * (r1 + r2), r1 + cos(psi) * (r1 + r2), 0])
v_OS2 = diff(r_OS2, alpha, 1) * alpha_dot + diff(r_OS2, psi, 1) * psi_dot
om2 = Matrix([0, 0, beta_dot])
r_S2P2 = Matrix([r2 * sin(psi), -r2 * cos(psi), 0])

v_P2 = v_OS2 + om2.cross(r_S2P2)

constraint1 = simplify(solve([v_P1[0] - v_P2[0]], [alpha_dot]))
constraint2 = simplify(solve([v_P1[1] - v_P2[1]], [alpha_dot]))

# print(constraint1 == constraint2)
# print(constraint1[alpha_dot])

# calculate Jacobians
v_OS1 = v_OS1.subs(alpha_dot, constraint1[alpha_dot])
om1 = om1.subs(alpha_dot, constraint1[alpha_dot])
v_OS2 = v_OS2.subs(alpha_dot, constraint1[alpha_dot])


r_S2S3 = Matrix([l * sin(phi), -l * cos(phi), 0])
om3 = Matrix([0, 0, phi_dot])
v_OS3 = v_OS2 + om3.cross(r_S2S3)

v_i = [v_OS1, v_OS2, v_OS3]
om_i = [om1, om2, om3]

J_i = [v.jacobian(omega) for v in v_i]
JR_i = [om.jacobian(omega) for om in om_i]

# Forces
M2 = Matrix([0, 0, -T])
M3 = Matrix([0, 0, T])

F_i = [Matrix([0, -mi * g, 0]) for mi in m]
M_i = [Matrix([0, 0, 0]), M2, M3]

# Impulse
p_i = [m[i] * v_i[i] for i in range(3)]
p_dot_i = [p.jacobian(omega) * omega_dot + p.jacobian(ang) * omega for p in p_i]

# Spin
NS_i = [theta[i] * om_i[i] for i in range(3)]
NS_dot_i = [NS.jacobian(omega) * omega_dot for NS in NS_i]

# dynamics
dyn = Matrix([0, 0, 0])
for i in range(3):
    dyn += simplify(J_i[i].T * (p_dot_i[i] - F_i[i])) + simplify(JR_i[i].T * (NS_dot_i[i] - M_i[i]))

# eliminate torque T by inspection
dyn_new = Matrix([0, 0, 0])
dyn_new[0] = dyn[0] + dyn[1]
dyn_new[1] = dyn[2]

# add motor dynamics
gamma_ddot = phi_ddot - beta_ddot
gamma_dot = phi_dot - beta_dot

dyn_new[2] = gamma_ddot - 1 / tau * (omega_cmd - gamma_dot)

A = simplify(dyn_new.jacobian(omega_dot))

sub_list = [(x, 'self.p.' + x)
            for x in ['g', 'l', 'm1', 'm2', 'm3', 'r1', 'r2', 'tau', 'theta1', 'theta2', 'theta3']]
for row in range(A.rows):
    for col in range(A.cols):
        if row > col and simplify(A[row, col] - A[col, row]) == 0:
            print('A[{},{}]=A[{},{}]'.format(row, col, col, row))
        else:
            print('A[{},{}] = {}'.format(row, col, A[row, col].subs(sub_list)))

b = simplify(dyn_new - A * omega_dot)
for row in range(b.rows):
    print('b[{}] = {}'.format(row, -b[row].subs(sub_list)))
