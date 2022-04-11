"""Script to generate symbolic dynamics of 2D N Ball Balancer

Derivation of the rigid multi-body dynamics using the Projected Newton-Euler method.
"""
import argparse
from sympy import Matrix, cse, diff, factor, expand, simplify, solve, symbols, zeros, sin, cos


def print_symbolic(mat, name, sub_list, func=lambda x: x, ignore_symmetry=True):
    for row in range(mat.rows):
        for col in range(mat.cols):
            if not ignore_symmetry and row > col and simplify(mat[row, col] - mat[col, row]) == 0:
                print('{}[{},{}] = {}[{},{}]'.format(name, row, col, name, col, row))
            else:
                print('{}[{},{}] = {}'.format(name, row, col, func(mat[row, col]).subs(sub_list)))

# angles


N = 3
alpha = [symbols('alpha_{}'.format(i)) for i in range(N)]
alpha_dot = [symbols('alpha_dot_{}'.format(i)) for i in range(N)]
alpha_ddot = [symbols('alpha_ddot_{}'.format(i)) for i in range(N)]

psi = [symbols('psi_{}'.format(i)) for i in range(N - 1)]
psi_dot = [symbols('psi_dot_{}'.format(i)) for i in range(N - 1)]
psi_ddot = [symbols('psi_ddot_{}'.format(i)) for i in range(N - 1)]

phi, phi_dot, phi_ddot = symbols('phi phi_dot phi_ddot')

ang = Matrix([alpha[N - 1]] + psi + [phi])
omega = Matrix([alpha_dot[N - 1]] + psi_dot + [phi_dot])
omega_dot = Matrix([alpha_ddot[N - 1]] + psi_ddot + [phi_ddot])

# parameter
l, m_l, theta_l = symbols('l, m_l, theta_l')
m = [symbols('m_{}'.format(i)) for i in range(N)] + [m_l]
r = [symbols('r_{}'.format(i)) for i in range(N)]
theta = [symbols('theta_{}'.format(i)) for i in range(N)] + [theta_l]

# constants
g, tau = symbols('g tau')

# inputs
omega_cmd, T = symbols('omega_cmd T')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="generation of symbolic dynamics of 2D N Ball Balancer")
    parser.add_argument(
        "-d",
        "--disable-printing-dynamics",
        help="disable printing of common sub-expressions for dynamic model",
        action="store_true",
        default=False)
    args = parser.parse_args()

    if args.disable_printing_dynamics and not args.print_latex_expressions:
        print('Nothing to do: {} ! Exiting.'.format(args.__dict__))
        exit()

    # kinematic constraints: lower ball rolling on the ground
    x = -r[0] * alpha[0]
    x_dot = -r[0] * alpha_dot[0]

    r_OS_i = [Matrix([x, r[0], 0])]
    v_OS_i = [Matrix([x_dot, 0, 0])]
    omega_i = [Matrix([0, 0, a]) for a in alpha_dot + [phi_dot]]

    # kinematic constraints: upper balls rolling on lower balls
    for i in range(N - 1):
        j = i + 1

        e_SiSj = Matrix([-sin(psi[i]), cos(psi[i]), 0])

        r_SiPi = r[i] * e_SiSj
        v_SiPi = omega_i[i].cross(r_SiPi)

        r_SiSj = (r[i] + r[j]) * e_SiSj

        r_OS_i.append(r_OS_i[i] + r_SiSj)

        v_SiSj = r_SiSj.jacobian(alpha) * Matrix(alpha_dot) + r_SiSj.jacobian(psi) * Matrix(psi_dot)

        v_OS_i.append(v_OS_i[i] + v_SiSj)

        r_SjPj = -r[j] * e_SiSj
        v_SiPj = v_SiSj + omega_i[j].cross(r_SjPj)

        constraint1 = solve([v_SiPi[0] - v_SiPj[0]], [alpha_dot[i]])[alpha_dot[i]]
        constraint2 = solve([v_SiPi[1] - v_SiPj[1]], [alpha_dot[i]])[alpha_dot[i]]

        assert(simplify(constraint1 - constraint2) == 0)

        sub_list = [(alpha_dot[i], constraint1)]

        v_OS_i[i] = v_OS_i[i].subs(sub_list)
        v_OS_i[j] = v_OS_i[j].subs(sub_list)
        omega_i[i] = omega_i[i].subs(sub_list)

    r_SnSl = l * Matrix([sin(phi), -cos(phi), 0])
    v_OS_i.append(v_OS_i[-1] + omega_i[-1].cross(r_SnSl))

    # calculate Jacobians
    J_i = [v.jacobian(omega) for v in v_OS_i]
    JR_i = [om.jacobian(omega) for om in omega_i]

    # Forces
    M2 = Matrix([0, 0, -T])
    M3 = Matrix([0, 0, T])

    F_i = [Matrix([0, -mi * g, 0]) for mi in m]
    M_i = [Matrix([0, 0, 0])] * (N - 1) + [M2, M3]

    # Impulse
    p_i = [m[i] * v_OS_i[i] for i in range(N + 1)]
    p_dot_i = [p.jacobian(omega) * omega_dot + p.jacobian(ang) * omega for p in p_i]

    # Spin
    NS_i = [theta[i] * omega_i[i] for i in range(N + 1)]
    NS_dot_i = [NS.jacobian(omega) * omega_dot for NS in NS_i]

    # dynamics
    dyn = zeros(N + 1, 1)
    for i in range(N + 1):
        dyn += J_i[i].T * (p_dot_i[i] - F_i[i]) + JR_i[i].T * (NS_dot_i[i] - M_i[i])

    # eliminate torque T by inspection
    dyn_new = dyn
    dyn_new[0] = dyn_new[0] + dyn_new[-1]
    dyn_new[-1] = 0
    assert(dyn_new.diff(T) == zeros(N + 1, 1))

    # add motor dynamics
    gamma_ddot = phi_ddot - alpha_ddot[-1]
    gamma_dot = phi_dot - alpha_dot[-1]

    dyn_new[-1] = gamma_ddot - 1 / tau * (omega_cmd - gamma_dot)

    if not args.disable_printing_dynamics:
        A = dyn_new.jacobian(omega_dot)
        b = -dyn_new.subs([(x, 0) for x in omega_dot])

        common_sub_expr = cse([A, b])

        sub_list = [
            (x,
             symbols('self.p.' +
                     x)) for x in [
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

    if not args.disable_printing_dynamics:
        print(dyn_lin)

        # calculate contact forces
        F = [p_dot_i[-1] - F_i[-1]]
        for i in range(N - 1, -1, -1):
            F = [F[0] + p_dot_i[i] - F_i[i]] + F

        common_sub_expr = cse(F)

        sub_list = [
            (x,
             symbols('self.p.' +
                     x)) for x in [
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

        print_symbolic(common_sub_expr[1][0], 'F1', sub_list)
        print_symbolic(common_sub_expr[1][1], 'F12', sub_list)
        print_symbolic(common_sub_expr[1][2], 'F23', sub_list)
