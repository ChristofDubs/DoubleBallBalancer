"""Script to generate symbolic dynamics of 2D Double Ball Balancer

Derivation of the rigid multi-body dynamics using the Projected Newton-Euler method.
"""
from sympy import *

print_python_expressions = True
print_latex_expressions = False


def print_symbolic(mat, name, sub_list, func=lambda x: x, ignore_symmetry=True):
    for row in range(mat.rows):
        for col in range(mat.cols):
            if not ignore_symmetry and row > col and simplify(mat[row, col] - mat[col, row]) == 0:
                print('{}[{},{}] = {}[{},{}]'.format(name, row, col, name, col, row))
            else:
                print('{}[{},{}] = {}'.format(name, row, col, func(mat[row, col]).subs(sub_list)))


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
r_S1P1 = r1 * Matrix([- sin(psi), cos(psi), 0])
v_OS1 = Matrix([-r1 * alpha_dot, 0, 0])

v_P1 = v_OS1 + om1.cross(r_S1P1)

r_OS2 = Matrix([x - sin(psi) * (r1 + r2), r1 + cos(psi) * (r1 + r2), 0])
v_OS2 = diff(r_OS2, alpha, 1) * alpha_dot + diff(r_OS2, psi, 1) * psi_dot
om2 = Matrix([0, 0, beta_dot])
r_S2P2 = Matrix([r2 * sin(psi), -r2 * cos(psi), 0])

v_P2 = v_OS2 + om2.cross(r_S2P2)

constraint1 = solve([v_P1[0] - v_P2[0]], [alpha_dot])
constraint2 = solve([v_P1[1] - v_P2[1]], [alpha_dot])

# print(constraint1 == constraint2)
# print(constraint1[alpha_dot])

# calculate Jacobians
v_OS1 = v_OS1.subs(alpha_dot, constraint1[alpha_dot])
om1 = om1.subs(alpha_dot, constraint1[alpha_dot])
v_OS2 = v_OS2.subs(alpha_dot, constraint1[alpha_dot])

r_S2S3 = l * Matrix([sin(phi), -cos(phi), 0])
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
    dyn += J_i[i].T * (p_dot_i[i] - F_i[i]) + JR_i[i].T * (NS_dot_i[i] - M_i[i])

if print_latex_expressions:
    A = dyn.jacobian(omega_dot)
    b = -dyn.subs([(x, 0) for x in omega_dot])

    latex_sub_list = [
        ('m1',
         'm_1'),
        ('m2',
         'm_2'),
        ('m3',
         'm_3'),
        ('r1',
         'r_1'),
        ('r2',
         'r_2'),
        ('tau',
         '\\tau'),
        ('theta1',
         '\\theta_1'),
        ('theta2',
         '\\theta_2'),
        ('theta3',
         '\\theta_3'),
        ('beta',
         '\\beta'),
        ('phi',
         '\\varphi'),
        ('psi',
         '\\psi'),
        ('beta_dot',
         '\\dot{\\beta}'),
        ('phi_dot',
         '\\dot{\\varphi}'),
        ('psi_dot',
         '\\dot{\\psi}')]

    print_symbolic(A, 'A', latex_sub_list, lambda x: factor(simplify(x)), False)
    print_symbolic(b, 'b', latex_sub_list, lambda x: simplify(factor(expand(x))))

# eliminate torque T by inspection
dyn_new = Matrix([0, 0, 0])
dyn_new[0] = dyn[0] + dyn[1]
dyn_new[2] = dyn[2]

# add motor dynamics
gamma_ddot = phi_ddot - beta_ddot
gamma_dot = phi_dot - beta_dot

dyn_new[1] = gamma_ddot - 1 / tau * (omega_cmd - gamma_dot)

if print_python_expressions:
    A = dyn_new.jacobian(omega_dot)
    b = -dyn_new.subs([(x, 0) for x in omega_dot])

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
            'theta3']]

    for term in common_sub_expr[0]:
        print('        {} = {}'.format(term[0], term[1].subs(sub_list)))

    print_symbolic(common_sub_expr[1][0], 'A', sub_list)
    print_symbolic(common_sub_expr[1][1], 'b', sub_list)

# linearize system around equilibrium [beta, 0, 0, 0, 0, 0]
eq = [
    (x,
     0) for x in [
        'phi',
        'psi',
        'beta_dot',
        'phi_dot',
        'psi_dot',
        'beta_dd',
        'phi_dd',
        'psi_dd',
        'omega_cmd']]

dyn_lin = dyn_new.subs(eq)
for vec in [ang, omega, omega_dot]:
    dyn_lin += dyn_new.jacobian(vec).subs(eq) * vec

dyn_lin += dyn_new.diff(omega_cmd, 1).subs(eq) * omega_cmd

if print_python_expressions:
    print(simplify(dyn_lin))

if print_latex_expressions:
    M = dyn_lin.jacobian(omega_dot)
    D = dyn_lin.jacobian(omega)
    K = dyn_lin.jacobian(ang)
    F = -dyn_new.diff(omega_cmd, 1)

    print_symbolic(M, 'M', latex_sub_list, lambda x: simplify(factor(expand(x))))
    print_symbolic(D, 'D', latex_sub_list, lambda x: factor(simplify(x)))
    print_symbolic(K, 'K', latex_sub_list, lambda x: factor(simplify(x)))
    print_symbolic(F, 'F', latex_sub_list, lambda x: factor(simplify(x)))
