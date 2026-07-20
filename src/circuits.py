import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm

L, J, H = 4, 1.0, 0.8  # qubits, coupling, transverse field


def _pstr(*ops_and_sites: tuple, n: int) -> str:
    chars = ["I"] * n
    for op, site in ops_and_sites:
        chars[n - 1 - site] = op
    return "".join(chars)


def tfim_h(n: int = L, j: float = J, h: float = H) -> SparsePauliOp:
    zz = [(_pstr(("Z", i), ("Z", i + 1), n=n), -j) for i in range(n - 1)]
    x = [(_pstr(("X", i), n=n), -h) for i in range(n)]
    return SparsePauliOp.from_list(zz + x)


def magnetization(n: int = L) -> SparsePauliOp:
    return SparsePauliOp.from_list([(_pstr(("Z", i), n=n), 1.0 / n) for i in range(n)])


def exact_magnetization(t: float, n: int = L, j: float = J, h: float = H) -> float:
    psi0 = Statevector.from_label("0" * n)
    psi_t = psi0.evolve(expm(-1j * tfim_h(n, j, h).to_matrix() * t))
    return float(np.real(psi_t.expectation_value(magnetization(n))))


def trotter_circuit(t: float, n_steps: int, n: int = L, j: float = J, h: float = H) -> QuantumCircuit:
    theta_zz, theta_x = 2 * j * t / n_steps, 2 * h * t / n_steps
    qc = QuantumCircuit(n)
    for _ in range(n_steps):
        for i in range(n - 1):
            qc.rzz(-theta_zz, i, i + 1)
        for i in range(n):
            qc.rx(-theta_x, i)
    return qc
