"""Script to generate symbolic dynamics of 2D N Ball Balancer

Derivation of the rigid multi-body dynamics using the Projected Newton-Euler method.
"""
import argparse
import pickle

from sympy import Matrix, cse, diff, factor, expand, simplify, solve, symbols, zeros, sin, cos

indent = " " * 4


def print_symbolic(file, mat, name, sub_list, ignore_symmetry=True):
    for row in range(mat.rows):
        for col in range(mat.cols):
            if not ignore_symmetry and row > col and simplify(mat[row, col] - mat[col, row]) == 0:
                file.write(indent + f'{name}[{row},{col}] = {name}[{col},{row}]\n')
            else:
                file.write(indent + f'{name}[{row},{col}] = {mat[row, col].subs(sub_list)}\n')


def writeCSE(expr_list: dict, file: str, state_dict: dict, sub_list: list, print_return: bool=True):
    common_sub_expr = cse(expr_list.values())

    for key, idx in state_dict.items():
        if not all((term.diff(key).is_zero_matrix for term in expr_list.values())):
            file.write(indent + f'{str(key)} = x[StateIndex.{idx}]\n')

    for term in common_sub_expr[0]:
        file.write(indent + f'{term[0]} = {term[1].subs(sub_list)}\n')

    for key, val in expr_list.items():
        file.write(indent + f'{key} = np.zeros({val.shape})\n')

    for i, term in enumerate(common_sub_expr[1]):
        print_symbolic(file, term, list(expr_list.keys())[i], sub_list)

    if print_return:
        file.write(indent + f'return [{", ".join(expr_list.keys())}]\n')


# angles
N = 2
alpha = [symbols('alpha_{}'.format(i)) for i in range(N)]
alpha_dot = [symbols('alpha_dot_{}'.format(i)) for i in range(N)]
alpha_ddot = [symbols('alpha_ddot_{}'.format(i)) for i in range(N)]

psi = [symbols('psi_{}'.format(i)) for i in range(N - 1)]
psi_dot = [symbols('psi_dot_{}'.format(i)) for i in range(N - 1)]
psi_ddot = [symbols('psi_ddot_{}'.format(i)) for i in range(N - 1)]

phi, phi_dot, phi_ddot = symbols('phi phi_dot phi_ddot')

ang = Matrix([alpha[N - 1]] + [phi] + psi)
omega = Matrix([alpha_dot[N - 1]] + [phi_dot] + psi_dot)
omega_dot = Matrix([alpha_ddot[N - 1]] + [phi_ddot] + psi_ddot)

# parameter
m = [symbols('m_{}'.format(i)) for i in range(N + 1)]
r = [symbols('r_{}'.format(i)) for i in range(N + 1)]
theta = [symbols('theta_{}'.format(i)) for i in range(N + 1)]

# constants
g, tau = symbols('g tau')

all_constants = m + r + theta + [g, tau]

# inputs
omega_cmd, T = symbols('omega_cmd T')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="generation of symbolic dynamics of 2D N Ball Balancer")

    parser.add_argument(
        "-d",
        "--disable-saving-dynamics",
        help="Disable writing non-linear dynamics to pickle file",
        action="store_true",
        default=False)
    parser.add_argument(
        "-p",
        "--print-dynamics",
        help="print common sub-expressions for dynamic model",
        action="store_true",
        default=True)

    args = parser.parse_args()

    if args.disable_saving_dynamics and not args.print_dynamics:
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

        omega_to_ang = [((alpha_dot + omega[1:])[i], (alpha + ang[1:])[i]) for i in range(len(alpha + ang[1:]))]
        sub_list = [(alpha_dot[i], constraint1), (alpha[i], constraint1.subs(omega_to_ang))]

        r_OS_i[i] = r_OS_i[i].subs(sub_list)
        r_OS_i[j] = r_OS_i[j].subs(sub_list)
        v_OS_i[i] = v_OS_i[i].subs(sub_list)
        v_OS_i[j] = v_OS_i[j].subs(sub_list)
        omega_i[i] = omega_i[i].subs(sub_list)

    r_SnSl = r[N] * Matrix([sin(phi), -cos(phi), 0])
    v_OS_i.append(v_OS_i[-1] + omega_i[-1].cross(r_SnSl))
    r_OS_i.append(r_OS_i[-1] + r_SnSl)

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
    dyn_new[0] = dyn_new[0] + dyn_new[1]
    dyn_new[1] = 0
    assert(dyn_new.diff(T) == zeros(N + 1, 1))

    # add motor dynamics
    gamma_ddot = phi_ddot - alpha_ddot[-1]
    gamma_dot = phi_dot - alpha_dot[-1]

    dyn_new[1] = gamma_ddot - 1 / tau * (omega_cmd - gamma_dot)

    # linearize system around equilibrium [alpha_N, 0, ... , 0]
    eq = [(x, 0) for x in ang[1:] + omega[:] + omega_dot[:] + [omega_cmd]]

    dyn_lin = dyn_new.subs(eq)
    for vec in [ang, omega, omega_dot]:
        dyn_lin += dyn_new.jacobian(vec).subs(eq) * vec

    dyn_lin += dyn_new.diff(omega_cmd, 1).subs(eq) * omega_cmd

    if not args.disable_saving_dynamics:
        dynamics_pickle_file = f'linear_dynamics_{N}.p'
        print(f'write dynamics to {dynamics_pickle_file}')
        pickle.dump(dyn_lin, open(dynamics_pickle_file, "wb"))

    if args.print_dynamics:
        file_name = f'dynamics_{N}.py'

        with open(file_name, 'w') as file:
            file.write("# Autogenerated using symbolic_dynamics_n.py; don't edit!\n")
            file.write('from dataclasses import dataclass\n')
            file.write('from enum import Enum\n')
            file.write('import numpy as np\n')
            file.write('from numpy import sin, cos\n')

            sub_list = [(x, symbols(f'param["{x}"]')) for x in all_constants]

            # enum with state indices
            state_dict = {x: str(x).upper() + '_IDX' for y in [ang, omega] for x in y}

            file.write('\n\nclass StateIndex(Enum):\n')
            for i, val in enumerate(state_dict.values()):
                file.write(indent + f'{val} = {i},\n')

            # compute omega dot
            A = dyn_new.jacobian(omega_dot)
            b = -dyn_new.subs([(x, 0) for x in omega_dot])
            file.write('\n\ndef computeOmegaDot(x, param, omega_cmd):\n')

            writeCSE({"A": A, "b": b}, file, state_dict, sub_list, False)

            file.write(indent + 'return np.linalg.solve(A, b)\n')

            # compute contact forces
            F = {f'F_{N}': p_dot_i[-1] - F_i[-1]}
            for i in range(N - 1, -1, -1):
                F[f'F_{i}'] = F[f'F_{i+1}'] + p_dot_i[i] - F_i[i]

            F = {key: F[key] for key in sorted(F.keys())}

            file.write('\n\ndef computeContactForces(x, param, omega_cmd):\n')

            file.write(indent + 'omega_dot = computeOmegaDot(x, param, omega_cmd)\n')
            for i in range(len(omega_dot)):
                file.write(indent + f'{omega_dot[i]} = omega_dot[StateIndex.{state_dict[ang[i]]}]\n')

            writeCSE(F, file, state_dict, sub_list)

            # r_OS_i
            file.write('\n\ndef computePositions(x, param):\n')

            writeCSE({f'r_OS_{i}': val for i, val in enumerate(r_OS_i)}, file, state_dict, sub_list)

        print(f"written dynamics to {file_name}")
