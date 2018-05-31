"""Script to generate symbolic dynamics of 3D Double Ball Balancer

Derivation of the rigid multi-body dynamics using the Projected Newton-Euler method.
"""
from sympy import symbols, Matrix, sin, cos, solve, diff, eye, diag, zeros, cse
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3

# position
x, y = symbols('x y')
x_dot, y_dot = symbols('x_dot y_dot')

# angles
alpha_z, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z, psi_x, psi_y = symbols(
    'alpha_z beta_x beta_y beta_z phi_x phi_y phi_z psi_x psi_y')

# angular velocities
phi_x_dot, phi_y_dot, phi_z_dot = symbols('phi_x_dot phi_y_dot phi_z_dot')
psi_x_dot, psi_y_dot = symbols('psi_x_dot psi_y_dot')
w_1x, w_1y, w_1z = symbols('w_1x w_1y w_1z')
w_2x, w_2y, w_2z = symbols('w_2x w_2y w_2z')
w_3x, w_3y, w_3z = symbols('w_3x w_3y w_3z')

omega_1 = Matrix([w_1x, w_1y, w_1z])
omega_2 = Matrix([w_2x, w_2y, w_2z])
b_omega_3 = Matrix([w_3x, w_3y, w_3z])

# angular accelerations
w_1_dot_z, w_3_dot_x, w_3_dot_y, w_3_dot_z, psi_x_ddot, psi_y_ddot, w_2_dot_x, w_2_dot_y, w_2_dot_z = symbols(
    'w_1_dot_z w_3_dot_x w_3_dot_y w_3_dot_z psi_x_ddot psi_y_ddot w_2_dot_x w_2_dot_y w_2_dot_z')

# parameter
l, m1, m2, m3, r1, r2, theta1, theta2, theta3x, theta3y, theta3z = symbols(
    'l m1 m2 m3 r1 r2 theta1 theta2 theta3x theta3y theta3z')

# constants
g = symbols('g')

# inputs
T = Matrix(symbols('Tx Ty Tz'))

# parameter lists:
m = [m1, m2, m3]
theta = [theta1 * eye(3), theta2 * eye(3), diag(theta3x, theta3y, theta3z)]

# kinematic constraints: lower ball rolling on the ground
r_OS1 = Matrix([x, y, r1])
v_OS1 = omega_1.cross(Matrix([0, 0, r1]))
x_dot = v_OS1[0]
y_dot = v_OS1[1]

# kinematic constraints: upper ball rolling on lower ball
e_S1S2 = rot_axis2(-psi_y) * rot_axis1(-psi_x) * Matrix([0, 0, 1])

r_S1P1 = r1 * e_S1S2

v_P1 = v_OS1 + omega_1.cross(r_S1P1)

r_S1S2 = (r1 + r2) * e_S1S2

r_OS2 = r_OS1 + r_S1S2

v_OS2 = diff(r_OS2, x, 1) * x_dot + diff(r_OS2, y, 1) * y_dot + \
    diff(r_OS2, psi_x, 1) * psi_x_dot + diff(r_OS2, psi_y, 1) * psi_y_dot

r_S2P2 = -r2 * e_S1S2

v_P2 = v_OS2 + omega_2.cross(r_S2P2)

constraints = v_P1 - v_P2

sol = solve(constraints, omega_1)

omega_1[0] = sol[w_1x]
omega_1[1] = sol[w_1y]
omega_1[2] = w_1z

sub_list = [(w_1x, omega_1[0]), (w_1y, omega_1[1]), (w_1z, omega_1[2])]

# eliminate omega_1
v_OS1 = v_OS1.subs(sub_list)
v_OS2 = v_OS2.subs(sub_list)

# lever arm
R_IB3 = rot_axis3(-phi_z) * rot_axis2(-phi_y) * rot_axis1(-phi_x)
r_S2S3 = R_IB3 * Matrix([0, 0, -l])
b_om_3 = Matrix([phi_x_dot, 0, 0]) + rot_axis1(phi_x) * Matrix([0, phi_y_dot, 0]) + \
    rot_axis1(phi_x) * rot_axis2(phi_y) * Matrix([0, 0, phi_z_dot])
jac = b_om_3.jacobian(Matrix([phi_x_dot, phi_y_dot, phi_z_dot]))
[phi_x_dot, phi_y_dot, phi_z_dot] = jac.LUsolve(b_omega_3)
v_OS3 = v_OS2 + R_IB3 * (b_omega_3.cross(Matrix([0, 0, -l])))


# calculate Jacobians
v_i = [v_OS1, v_OS2, v_OS3]
om_i = [omega_1, omega_2, b_omega_3]

omega_dot = Matrix([w_1_dot_z, psi_x_ddot, psi_y_ddot, w_2_dot_x,
                    w_2_dot_y, w_2_dot_z, w_3_dot_x, w_3_dot_y, w_3_dot_z])
omega = Matrix([w_1z, psi_x_dot, psi_y_dot, w_2x, w_2y, w_2z, w_3x, w_3y, w_3z])
ang = Matrix([alpha_z, psi_x, psi_y, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z])
ang_dot = Matrix([w_1z, psi_x_dot, psi_y_dot, w_2x, w_2y, w_2z, phi_x_dot, phi_y_dot, phi_z_dot])

J_i = [v.jacobian(omega) for v in v_i]
JR_i = [om.jacobian(omega) for om in om_i]

# Forces
F_i = [Matrix([0, 0, -mi * g]) for mi in m]
M_i = [Matrix([0, 0, 0]), -T, R_IB3.T * T]

# Impulse
p_i = [m[i] * v_i[i] for i in range(3)]
p_dot_i = [p.jacobian(omega) * omega_dot + p.jacobian(ang) * ang_dot for p in p_i]

# Spin
omega_diff_i = [
    om_i[i].jacobian(omega) *
    omega_dot +
    om_i[i].jacobian(ang) *
    ang_dot for i in range(3)]
NS_dot_i = [
    theta[i] *
    omega_diff_i[i] +
    om_i[i].cross(
        theta[i] *
        om_i[i]) for i in range(3)]

# dynamics
print('generating dynamics')
dyn = zeros(9, 1)
for i in range(3):
    dyn += J_i[i].T * (p_dot_i[i] - F_i[i]) + JR_i[i].T * (NS_dot_i[i] - M_i[i])
    print('generated term {} of 3 dynamic terms'.format(i))

A = dyn.jacobian(omega_dot)
b = dyn.subs([(x, 0) for x in omega_dot])

common_sub_expr = cse([A, b])

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
        'theta3x',
        'theta3y',
        'theta3z']]

for term in common_sub_expr[0]:
    print('        {} = {}'.format(term[0], term[1].subs(sub_list)))

for row in range(A.rows):
    for col in range(A.cols):
        print('        A[{},{}] = {}'.format(row, col,
                                             common_sub_expr[1][0][row, col].subs(sub_list)))

for row in range(b.rows):
    print('        b[{}] = {}'.format(row, -common_sub_expr[1][1][row].subs(sub_list)))

# kinematic relations
common_sub_expr = cse(omega_1)
for term in common_sub_expr[0]:
    print('        {} = {}'.format(term[0], term[1].subs(sub_list)))

for row in range(omega_1.rows):
    print('        omega_1[{}] = {}'.format(row, common_sub_expr[1][0][row].subs(sub_list)))

# position vectors
common_sub_expr = cse([r_S1S2, r_S2S3])
for term in common_sub_expr[0]:
    print('        {} = {}'.format(term[0], term[1].subs(sub_list)))

for row in range(r_S1S2.rows):
    print('        r_S1S2[{}] = {}'.format(row, common_sub_expr[1][0][row].subs(sub_list)))

for row in range(r_S2S3.rows):
    print('        r_S2S3[{}] = {}'.format(row, common_sub_expr[1][1][row].subs(sub_list)))
