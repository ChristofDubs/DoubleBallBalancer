"""Script to generate symbolic dynamics of 3D Double Ball Balancer

Derivation of the rigid multi-body dynamics using the Projected Newton-Euler method.
"""
import argparse
import pickle

from sympy import symbols, Matrix, solve, diff, eye, diag, zeros, cse, pi, exp, Max
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3


def print_common_terms(mat, sub_list):
    for term in mat:
        print('{} = {}'.format(term[0], term[1].subs(sub_list)))


def print_symbolic(mat, name, sub_list):
    for row in range(mat.rows):
        if mat.cols == 1:
            print('{}[{}] = {}'.format(name, row, mat[row, 0].subs(sub_list)))
        else:
            for col in range(mat.cols):
                print('{}[{},{}] = {}'.format(name, row, col, mat[row, col].subs(sub_list)))


# position
x, y = symbols('x y')

# angles
alpha_z, beta_x, beta_y, beta_z, phi_x, phi_y, psi_x, psi_y = symbols(
    'alpha_z beta_x beta_y beta_z phi_x phi_y psi_x psi_y')

ang = Matrix([alpha_z, psi_x, psi_y, beta_x, beta_y, beta_z, phi_x, phi_y])

# angular velocities
phi_x_dot, phi_y_dot = symbols('phi_x_dot phi_y_dot')
beta_x_dot, beta_y_dot, beta_z_dot = symbols('beta_x_dot beta_y_dot beta_z_dot')
psi_x_dot, psi_y_dot = symbols('psi_x_dot psi_y_dot')
w_1x, w_1y, w_1z = symbols('w_1x w_1y w_1z')
w_2x, w_2y, w_2z = symbols('w_2x w_2y w_2z')

omega_1 = Matrix([w_1x, w_1y, w_1z])
b_omega_2 = Matrix([w_2x, w_2y, w_2z])

omega = Matrix([w_1z, psi_x_dot, psi_y_dot, w_2x, w_2y, w_2z, phi_x_dot, phi_y_dot])

# angular accelerations
w_1_dot_z, phi_x_ddot, phi_y_ddot, psi_x_ddot, psi_y_ddot, w_2_dot_x, w_2_dot_y, w_2_dot_z = symbols(
    'w_1_dot_z phi_x_ddot phi_y_ddot psi_x_ddot psi_y_ddot w_2_dot_x w_2_dot_y w_2_dot_z')

omega_dot = Matrix([w_1_dot_z, psi_x_ddot, psi_y_ddot, w_2_dot_x,
                    w_2_dot_y, w_2_dot_z, phi_x_ddot, phi_y_ddot])

# parameter
a, l, m1, m2, m3, mu1, mu12, r1, r2, tau, theta1, theta2, theta3x, theta3y, theta3z = symbols(
    'a l m1 m2 m3 mu1 mu12 r1 r2 tau theta1 theta2 theta3x theta3y theta3z')

# constants
g = symbols('g')

# inputs
Tx, Ty = symbols('Tx Ty')
b_T = Matrix([Tx, Ty, 0])
omega_x_cmd, omega_y_cmd = symbols('omega_x_cmd omega_y_cmd')
omega_cmd = Matrix([omega_x_cmd, omega_y_cmd])

# parameter lists:
m = [m1, m2, m3]
theta = [theta1 * eye(3), theta2 * eye(3), diag(theta3x, theta3y, theta3z)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="generation of symbolic dynamics of 3D Double Ball Balancer")
    parser.add_argument(
        "-d",
        "--save-dynamics",
        help="Write non-linear dynamics to pickle file",
        action="store_true",
        default=True)
    parser.add_argument(
        "-p",
        "--print-dynamics",
        help="print common sub-expressions for dynamic model",
        action="store_true")
    parser.add_argument(
        "-l",
        "--save-linear",
        help="Write linear dynamics to pickle file",
        action="store_true")
    args = parser.parse_args()

    if not (args.save_dynamics or args.print_dynamics or args.save_linear):
        print('Nothing to do: {} ! Exiting.'.format(args.__dict__))
        exit()

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

    R_IB2 = rot_axis3(-beta_z) * rot_axis2(-beta_y) * rot_axis1(-beta_x)

    v_P2 = v_OS2 + (R_IB2 * b_omega_2).cross(r_S2P2)

    constraints = v_P1 - v_P2

    sol = solve(constraints, omega_1)

    omega_1[0] = sol[w_1x]
    omega_1[1] = sol[w_1y]
    omega_1[2] = w_1z

    sub_list = [(w_1x, omega_1[0]), (w_1y, omega_1[1]), (w_1z, omega_1[2])]

    # eliminate omega_1
    v_OS1 = v_OS1.subs(sub_list)
    v_OS2 = v_OS2.subs(sub_list)

    # upper ball
    R_IB2 = rot_axis3(-beta_z) * rot_axis2(-beta_y) * rot_axis1(-beta_x)
    b_om_2 = Matrix([beta_x_dot, 0, 0]) + rot_axis1(beta_x) * Matrix([0, beta_y_dot, 0]
                                                                     ) + rot_axis1(beta_x) * rot_axis2(beta_y) * Matrix([0, 0, beta_z_dot])
    jac = b_om_2.jacobian(Matrix([beta_x_dot, beta_y_dot, beta_z_dot]))
    [beta_x_dot, beta_y_dot, beta_z_dot] = jac.LUsolve(b_omega_2)

    # lever arm
    R_B2B3 = rot_axis2(-phi_y) * rot_axis1(-phi_x)
    R_IB3 = R_IB2 * R_B2B3
    r_S2S3 = R_IB3 * Matrix([0, 0, -l])
    b_omega_3 = Matrix([phi_x_dot, 0, 0]) + rot_axis1(phi_x) * \
        Matrix([0, phi_y_dot, 0]) + R_B2B3.T * b_omega_2

    v_OS3 = v_OS2 + R_IB3 * (b_omega_3.cross(Matrix([0, 0, -l])))

    # calculate Jacobians
    v_i = [v_OS1, v_OS2, v_OS3]
    om_i = [omega_1, b_omega_2, b_omega_3]

    ang_dot = Matrix([w_1z, psi_x_dot, psi_y_dot, beta_x_dot,
                      beta_y_dot, beta_z_dot, phi_x_dot, phi_y_dot])

    J_i = [v.jacobian(omega) for v in v_i]
    JR_i = [om.jacobian(omega) for om in om_i]

    # Impulse
    p_i = [m[i] * v_i[i] for i in range(3)]
    p_dot_i = [p.jacobian(omega) * omega_dot + p.jacobian(ang) * ang_dot for p in p_i]

    # Forces
    F_i = [Matrix([0, 0, -mi * g]) for mi in m]

    F23 = p_dot_i[2] - F_i[2]
    F12 = p_dot_i[1] - F_i[1] + F23
    F1 = p_dot_i[0] - F_i[0] + F12

    # torsional friction model: http://gazebosim.org/tutorials?tut=torsional_friction&cat=physics
    # smooth version of sign, to avoid numerical problems due to sign's discontinuity
    def sign(x): return 2 / (1 + exp(-x)) - 1

    f1_scale = 3 * pi / 16 * (a * r1) * mu1 * sign(-w_1z)

    w_21 = ((R_IB2 * b_omega_2) - omega_1).dot(e_S1S2)

    f12_scale = 3 * pi / 16 * (a * Max(r1, r2)) * mu12 * sign(-w_21)

    M1 = f1_scale * F1[2] * Matrix([0, 0, 1])
    M12 = f12_scale * F12.dot(e_S1S2) * e_S1S2

    M_i = [Matrix([0, 0, 0]) + M1 - M12, R_IB2.T * M12 - R_B2B3 * b_T, b_T]

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
    dyn = zeros(8, 1)
    for i in range(3):
        dyn += J_i[i].T * (p_dot_i[i] - F_i[i]) + JR_i[i].T * (NS_dot_i[i] - M_i[i])
        print('generated term {} of 3 dynamic terms'.format(i))

    # replace the last 2 equations (the only ones containing T)
    dyn[6] = phi_x_ddot - 1 / tau * (omega_x_cmd - phi_x_dot)
    dyn[7] = phi_y_ddot - 1 / tau * (omega_y_cmd - phi_y_dot)

    # check that all Tx, Ty terms are eliminated
    print('all T terms eliminated: {}'.format(
        Matrix(dyn[:]).jacobian(Matrix([Tx, Ty])) == zeros(8, 2)))

    # set Tx, Ty to zero directly instead of simplifying (terms can be ... + Tx + ... - Tx)
    dyn = dyn.subs([('Tx', 0), ('Ty', 0)])

    if args.save_dynamics:
        dynamics_pickle_file = 'dynamics.p'
        print('write dynamics to {}'.format(dynamics_pickle_file))
        pickle.dump(dyn, open(dynamics_pickle_file, "wb"))

    if args.print_dynamics:

        A = dyn.jacobian(omega_dot)
        b = -dyn.subs([(x, 0) for x in omega_dot])

        common_sub_expr = cse([A, b])

        sub_list = [
            (x,
             'self.p.' +
             x) for x in [
                'a',
                'g',
                'l',
                'm1',
                'm2',
                'm3',
                'mu1',
                'mu12',
                'r1',
                'r2',
                'tau',
                'theta1',
                'theta2',
                'theta3x',
                'theta3y',
                'theta3z']]

        print_common_terms(common_sub_expr[0], sub_list)
        print_symbolic(common_sub_expr[1][0], 'A', sub_list)
        print_symbolic(common_sub_expr[1][1], 'b', sub_list)

        # kinematic relations
        common_sub_expr = cse(omega_1)

        print_common_terms(common_sub_expr[0], sub_list)
        print_symbolic(common_sub_expr[1][0], 'omega_1', sub_list)

        # position vectors
        common_sub_expr = cse([r_S1S2, r_S2S3])

        print_common_terms(common_sub_expr[0], sub_list)
        print_symbolic(common_sub_expr[1][0], 'r_S1S2', sub_list)
        print_symbolic(common_sub_expr[1][1], 'r_S2S3', sub_list)

    if args.save_linear:

        # linearize system around equilibrium [0, ... , 0, x, y]
        eq = [(x, 0) for x in omega_dot]
        eq.extend([(x, 0) for x in omega])
        eq.extend([(x, 0) for x in ang])
        eq.extend([(x, 0) for x in omega_cmd])

        dyn_lin = dyn.subs(eq)
        for vec in [ang, omega, omega_dot, omega_cmd]:
            dyn_lin += dyn.jacobian(vec).subs(eq) * vec

        pickle_file = 'linear_dynamics.p'
        print('write dynamics to {}'.format(pickle_file))
        pickle.dump(dyn_lin, open(pickle_file, "wb"))
