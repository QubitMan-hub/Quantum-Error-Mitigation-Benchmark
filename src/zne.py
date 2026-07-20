import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


def fold_circuit(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be a positive odd integer (1, 3, 5, ...)")
    folded = qc.copy()
    for _ in range((scale - 1) // 2):
        folded.compose(qc.inverse(), inplace=True)
        folded.compose(qc, inplace=True)
    return folded


def _z_parity(label: str, bits: str) -> int:
    ones = sum(int(b) for c, b in zip(label, bits) if c == "Z")
    return 1 - 2 * (ones % 2)


def noisy_expectation(qc: QuantumCircuit, obs: SparsePauliOp, noise_model: NoiseModel, shots: int = 4096) -> float:
    sim = AerSimulator(noise_model=noise_model, method="density_matrix")

    if not shots or shots <= 0:
        circ = qc.copy()
        circ.save_density_matrix()
        dm = sim.run(circ, shots=1, optimization_level=0).result().data(0)["density_matrix"]
        return float(dm.expectation_value(obs).real)

    circ = qc.copy()
    circ.measure_all()
    counts = sim.run(circ, shots=shots, optimization_level=0).result().get_counts()
    counts = {bs.replace(" ", ""): c for bs, c in counts.items()}

    ev = 0.0
    for pauli, coeff in zip(obs.paulis, obs.coeffs):
        label = pauli.to_label()
        if any(c not in "ZI" for c in label):
            raise ValueError("noisy_expectation only supports Z-diagonal observables")
        term = sum(_z_parity(label, bs) * c for bs, c in counts.items())
        ev += float(np.real(coeff)) * term / shots
    return ev


def zne_expectation(qc: QuantumCircuit, obs: SparsePauliOp, noise_model: NoiseModel,
                     scales: tuple = (1, 3, 5), shots: int = 4096, return_raw: bool = False):
    measured = [noisy_expectation(fold_circuit(qc, s), obs, noise_model, shots) for s in scales]
    coeffs = np.polyfit(scales, measured, deg=len(scales) - 1)  # Richardson extrapolation to zero noise
    zne_value = float(np.polyval(coeffs, 0))
    return (zne_value, list(zip(scales, measured))) if return_raw else zne_value
