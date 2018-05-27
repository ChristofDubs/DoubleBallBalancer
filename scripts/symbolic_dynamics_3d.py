"""Script to generate symbolic dynamics of 3D Double Ball Balancer

Derivation of the rigid multi-body dynamics using the Projected Newton-Euler method.
"""
from sympy import symbols, Matrix, sin, cos, simplify, solve, diff, eye, diag, zeros
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3

# position
x, y = symbols('x y')
x_dot, y_dot = symbols('x_dot y_dot')

# angles
beta_x, beta_y, beta_z, phi_x, phi_y, phi_z, psi_y, psi_z = symbols(
    'beta_x, beta_y, beta_z phi_x phi_y phi_z psi_y psi_z')

# angular velocities
phi_x_dot, phi_y_dot, phi_z_dot = symbols('phi_x_dot phi_y_dot phi_z_dot')
psi_y_dot, psi_z_dot = symbols('psi_y_dot psi_z_dot')
w_1x, w_1y, w_1z = symbols('w_1x w_1y w_1z')
w_2x, w_2y, w_2z = symbols('w_2x w_2y w_2z')

omega_1 = Matrix([w_1x, w_1y, w_1z])
omega_2 = Matrix([w_2x, w_2y, w_2z])

# angular accelerations
phi_x_ddot, phi_y_ddot, phi_z_ddot, psi_y_ddot, psi_z_ddot, w_2_dot_x, w_2_dot_y, w_2_dot_z = symbols(
    'phi_x_ddot phi_y_ddot phi_z_ddot psi_y_ddot psi_z_ddot w_2_dot_x w_2_dot_y w_2_dot_z')

# parameter
l, m1, m2, m3, r1, r2, theta1, theta2, theta3x, theta3y, theta3z = symbols(
    'l m1 m2 m3 r1 r2 theta1 theta2 theta3x theta3y theta3z')

# constants
g = symbols('g')

# inputs
b_T = Matrix(symbols('Tx Ty Tz'))

# parameter lists:
m = [m1, m2, m3]
theta = [theta1 * eye(3), theta2 * eye(3), diag(theta3x, theta3y, theta3z)]

# kinematic constraints: lower ball rolling on the ground
r_OS1 = Matrix([x, y, r1])
v_OS1 = omega_1.cross(Matrix([0, 0, r1]))
x_dot = v_OS1[0]
y_dot = v_OS1[1]

# kinematic constraints: upper ball rolling on lower ball
e_S1S2 = Matrix([cos(psi_z) * sin(psi_y), sin(psi_z) * sin(psi_y), cos(psi_y)])

r_S1P1 = r1 * e_S1S2

v_P1 = v_OS1 + omega_1.cross(r_S1P1)

r_S1S2 = (r1 + r2) * e_S1S2

r_OS2 = r_OS1 + r_S1S2

v_OS2 = diff(r_OS2, x, 1) * x_dot + diff(r_OS2, y, 1) * y_dot + \
    diff(r_OS2, psi_y, 1) * psi_y_dot + diff(r_OS2, psi_z, 1) * psi_z_dot

r_S2P2 = -r2 * e_S1S2

v_P2 = v_OS2 + omega_2.cross(r_S2P2)

constraints = simplify(v_P1 - v_P2)

sol = solve(constraints, omega_1)

# somehow, w_1z is not in the sol dict... need to back-substitute sol for w_1x and solve again.
omega_1[2] = simplify(solve(constraints[1].subs(w_1x, sol[w_1x]), w_1z)[0])
omega_1[0] = simplify(sol[w_1x].subs(w_1z, omega_1[2]))
omega_1[1] = simplify(sol[w_1y].subs(w_1z, omega_1[2]))

sub_list = [(w_1x, omega_1[0]), (w_1y, omega_1[1]), (w_1z, omega_1[2])]

# eliminate omega_1
v_OS1 = simplify(v_OS1.subs(sub_list))
v_OS2 = simplify(v_OS2.subs(sub_list))

# lever arm
R_IB3 = rot_axis3(-phi_z) * rot_axis2(-phi_y) * rot_axis1(-phi_x)
b_omega_3 = Matrix([phi_x_dot, 0, 0]) + rot_axis1(phi_x) * Matrix([0, phi_y_dot, 0]
                                                                  ) + rot_axis1(phi_x) * rot_axis2(phi_y) * Matrix([0, 0, phi_z_dot])
# print(b_omega_3.jacobian(Matrix([phi_x_dot, phi_y_dot, phi_z_dot])))
# i_omega_3 = simplify(R_IB3 * b_omega_3)
v_OS3 = v_OS2 + simplify(R_IB3 * b_omega_3.cross(Matrix([0, 0, -l])))


# calculate Jacobians
v_i = [v_OS1, v_OS2, v_OS3]
om_i = [omega_1, omega_2, b_omega_3]

omega_dot = Matrix([phi_x_ddot, phi_y_ddot, phi_z_ddot, psi_y_ddot,
                    psi_z_ddot, w_2_dot_x, w_2_dot_y, w_2_dot_z])
omega = Matrix([phi_x_dot, phi_y_dot, phi_z_dot, psi_y_dot, psi_z_dot, w_2x, w_2y, w_2z])
ang = Matrix([phi_x, phi_y, phi_z, psi_y, psi_z, beta_x, beta_y, beta_z])

J_i = [v.jacobian(omega) for v in v_i]
JR_i = [om.jacobian(omega) for om in om_i]

# Forces
F_i = [Matrix([0, 0, -mi * g]) for mi in m]
M_i = [Matrix([0, 0, 0]), -R_IB3 * b_T, b_T]

# Impulse
p_i = [m[i] * v_i[i] for i in range(3)]
p_dot_i = [p.jacobian(omega) * omega_dot + p.jacobian(ang) * omega for p in p_i]

# Spin
omega_diff_i = [
    om_i[i].jacobian(omega) *
    omega_dot +
    om_i[i].jacobian(ang) *
    omega for i in range(3)]
NS_dot_i = [theta[i] * omega_diff_i[i] + om_i[i].cross(theta[i] * om_i[i]) for i in range(3)]

# dynamics
dyn = zeros(8, 1)
for i in range(3):
    dyn += simplify(J_i[i].T * (p_dot_i[i] - F_i[i])) + simplify(JR_i[i].T * (NS_dot_i[i] - M_i[i]))

A = simplify(dyn.jacobian(omega_dot))

sub_list = [
    (x,
     'self.p.' +
     x) for x in [
        'g',
        'l',
        'm1',
        'm2',
        'm3',
        'r1',
        'r2',
        'tau',
        'theta1',
        'theta2',
        'theta3',
        'theta3x',
        'theta3y',
        'theta3z']]
for row in range(A.rows):
    for col in range(A.cols):
        if row > col and simplify(A[row, col] - A[col, row]) == 0:
            print('A[{},{}] = A[{},{}]'.format(row, col, col, row))
        else:
            print('A[{},{}] = {}'.format(row, col, A[row, col].subs(sub_list)))

b = simplify(dyn - A * omega_dot)
for row in range(b.rows):
    print('b[{}] = {}'.format(row, -b[row].subs(sub_list)))

# verify term separation
dyn2 = A * omega_dot + b

print([simplify(dyn[i] - dyn2[i]) == 0 for i in range(5)])

# kinematic relations
for i, om in enumerate(omega_1):
    print('omega_1[{}] = {}'.format(i, om.subs(sub_list)))
