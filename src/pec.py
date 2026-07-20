import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel

from src.zne import noisy_expectation

_GATE = {"X": QuantumCircuit.x, "Y": QuantumCircuit.y, "Z": QuantumCircuit.z}


def _quasi_probs(p: float) -> dict:
    denom = 4 - 4 * p
    return {"I": 1 + 3 * p / denom, "X": -p / denom, "Y": -p / denom, "Z": -p / denom}


def pec_expectation(qc: QuantumCircuit, obs: SparsePauliOp, noise_model: NoiseModel, one_qubit_error: float,
                     n_samples: int = 200, shots_per_sample: int = 512, seed: int = None) -> tuple:
    rng = np.random.default_rng(seed)
    probs = _quasi_probs(one_qubit_error)
    paulis = list(probs.keys())
    weights = np.array([probs[p] for p in paulis])
    norm = np.abs(weights).sum()
    sample_probs = np.abs(weights) / norm
    signs = np.sign(weights)

    n_locations = sum(instr.operation.name == "rx" for instr in qc.data)
    if n_locations == 0:
        raise ValueError("circuit has no 'rx' gates to correct")
    overhead = norm ** n_locations if n_locations < 200 else float("inf")

    results = []
    for _ in range(n_samples):
        corrected = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        sign_product = 1.0
        for instr in qc.data:
            corrected.append(instr.operation, instr.qubits, instr.clbits)
            if instr.operation.name == "rx":
                idx = rng.choice(len(paulis), p=sample_probs)
                pauli, sign = paulis[idx], signs[idx]
                sign_product *= sign
                if pauli != "I":
                    _GATE[pauli](corrected, instr.qubits[0])
        val = noisy_expectation(corrected, obs, noise_model, shots=shots_per_sample)
        results.append(sign_product * val)

    pec_value = float(np.mean(results)) * (norm ** n_locations)
    return pec_value, float(overhead)
