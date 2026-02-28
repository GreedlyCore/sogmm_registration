"""
validate_partial_wrt_u.py

Validates C++ partial_wrt_u() against SymPy symbolic derivatives.

R(u) is the rotation matrix derived from axis-angle u = [u1, u2, u3]
via the unit quaternion:
  theta = ||u||,  half = theta/2
  qw = cos(half),  q_vec = sin(half)/theta * u
  R_ij from quaternion multiplication formula

Derivatives:
  J[i]    = dR/du_i          (3x3)
  H[i][j] = d2R/(du_i du_j)  (3x3)
"""

import subprocess, sys, os
import numpy as np
import sympy as sp

# ── SymPy symbolic setup ─────────────────────────────────────────────────────

u1s, u2s, u3s = sp.symbols('u1 u2 u3', real=True)
u_syms = [u1s, u2s, u3s]

t6  = u1s**2 + u2s**2 + u3s**2   # ||u||^2
t7  = sp.sqrt(t6)                  # ||u||
half = t7 / 2
s2  = sp.sin(half)                 # sin(theta/2)
c2  = sp.cos(half)                 # cos(theta/2)

# unit quaternion components
qw = c2
qx = s2 / t7 * u1s
qy = s2 / t7 * u2s
qz = s2 / t7 * u3s

# rotation matrix from quaternion (standard formula)
R_sym = sp.Matrix([
    [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
    [  2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qw*qx)],
    [  2*(qx*qz - qw*qy),   2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
])

print("Computing symbolic Jacobians ...", flush=True)
dR_sym  = [sp.diff(R_sym, u) for u in u_syms]          # J[0..2]

print("Computing symbolic Hessians (slow) ...", flush=True)
d2R_sym = [[sp.diff(dR_sym[i], u_syms[j]) for j in range(3)]
           for i in range(3)]                            # H[i][j]

# lambdify: near-zero handled by passing exact float 0 avoidance in test points
_args = u_syms

J_fn  = [[sp.lambdify(_args, dR_sym[i][r, c], 'numpy')
          for r in range(3) for c in range(3)]
         for i in range(3)]

H_fn  = [[[sp.lambdify(_args, d2R_sym[i][j][r, c], 'numpy')
           for r in range(3) for c in range(3)]
          for j in range(3)]
         for i in range(3)]


def sympy_J(u):
    v = u.tolist()
    out = []
    for i in range(3):
        M = np.array([fn(*v) for fn in J_fn[i]], dtype=float).reshape(3, 3)
        out.append(M)
    return out


def sympy_H(u):
    v = u.tolist()
    out = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            M = np.array([fn(*v) for fn in H_fn[i][j]], dtype=float).reshape(3, 3)
            out[i][j] = M
    return out


# ── C++ runner ───────────────────────────────────────────────────────────────

# Locate binary: check build/install dirs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR  = os.path.join(_SCRIPT_DIR, 'wet/install/gmm_d2d_registration/lib/gmm_d2d_registration')
_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, 'wet/build/gmm_d2d_registration/test_partial_wrt_u'),
    os.path.join(_BUILD_DIR,  'test_partial_wrt_u'),
    # colcon install path
    os.path.expanduser('~/thesis/sogmm_registration/wet/build/gmm_d2d_registration/test_partial_wrt_u'),
    os.path.expanduser('~/thesis/sogmm_registration/wet/install/gmm_d2d_registration/lib/gmm_d2d_registration/test_partial_wrt_u'),
]
CPP_BIN = None
for c in _CANDIDATES:
    if os.path.isfile(c):
        CPP_BIN = c
        break


def cpp_JH(u):
    """Run C++ binary, parse stdout into J and H numpy arrays."""
    if CPP_BIN is None:
        return None, None
    result = subprocess.run(
        [CPP_BIN, str(u[0]), str(u[1]), str(u[2])],
        capture_output=True, text=True
    )
    J = [None]*3
    H = [[None]*3 for _ in range(3)]
    for line in result.stdout.splitlines():
        parts = line.split()
        label = parts[0]
        vals  = np.array(parts[1:], dtype=float).reshape(3, 3)
        if label.startswith('J'):
            J[int(label[1])] = vals
        elif label.startswith('H'):
            H[int(label[1])][int(label[2])] = vals
    return J, H


# ── Test points ──────────────────────────────────────────────────────────────

test_points = [
    np.array([0.3, 0.5, 0.7]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.5]),
    np.array([0.1, -0.2, 1.1]),
    np.array([1e-6, 0.0, 0.0]),       # near-zero: Taylor branch in C++
    np.array([1e-12, 1e-12, 1e-12]),  # deep near-zero
]

# ── Run validation ───────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SymPy vs C++ partial_wrt_u validation")
print("="*70)

if CPP_BIN is None:
    print("\n[WARN] C++ binary not found. Skipping numeric comparison.")
    print("       Build with colcon and re-run.")
    cpp_available = False
else:
    print(f"C++ binary: {CPP_BIN}")
    cpp_available = True

for u in test_points:
    norm = np.linalg.norm(u)
    print(f"\nu = {u}  (||u|| = {norm:.2e})")

    # SymPy evaluation (use small offset to avoid exact singularity at zero)
    if norm < 1e-11:
        # For deep near-zero, SymPy limit = use tiny perturbation
        u_eval = u + 1e-13 * np.ones(3)
    else:
        u_eval = u

    try:
        J_sp = sympy_J(u_eval)
        H_sp = sympy_H(u_eval)
    except Exception as e:
        print(f"  SymPy error: {e}")
        continue

    # skew-symmetry check: dR[i] @ R.T must be skew-symmetric
    R_val = np.array(R_sym.subs(list(zip(u_syms, u_eval.tolist()))).tolist(),
                     dtype=float)
    print("  Skew-symmetry check (dR[i] @ R.T + transpose should be ~0):")
    for i in range(3):
        A = J_sp[i] @ R_val.T
        skew_err = np.max(np.abs(A + A.T))
        status = "OK" if skew_err < 1e-6 else "FAIL"
        print(f"    J[{i}] @ R.T skew err = {skew_err:.2e}  [{status}]")

    if not cpp_available:
        continue

    J_cpp, H_cpp = cpp_JH(u)
    if J_cpp[0] is None:
        print("  C++ parse failed.")
        continue

    # Compare J
    print("  J comparison (SymPy vs C++):")
    for i in range(3):
        err = np.max(np.abs(J_sp[i] - J_cpp[i]))
        status = "OK" if err < 1e-4 else "FAIL"
        print(f"    J[{i}] max_err = {err:.2e}  [{status}]")

    # Compare H
    print("  H comparison (SymPy vs C++):")
    for i in range(3):
        for j in range(3):
            err = np.max(np.abs(H_sp[i][j] - H_cpp[i][j]))
            status = "OK" if err < 1e-4 else "FAIL"
            print(f"    H[{i}][{j}] max_err = {err:.2e}  [{status}]")

print("\n" + "="*70)
print("Done.")
