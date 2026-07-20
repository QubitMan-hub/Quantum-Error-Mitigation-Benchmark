from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def build_noise_model(two_qubit_error: float = 0.01, one_qubit_error: float = 0.001,
                       readout_error: float = 0.02) -> NoiseModel:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(one_qubit_error, 1), ["rx", "ry", "rz", "h", "x", "sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ["rzz", "cx", "cz"])
    ro = ReadoutError([[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])
    nm.add_all_qubit_readout_error(ro)
    return nm
