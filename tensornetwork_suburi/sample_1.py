import quimb as qu
import quimb.tensor as qtn

# 10 qubits and tag the initial wavefunction tensors
circ = qtn.Circuit(N=10, tags='PSI0')

# initial layer of hadamards
for i in range(10):
    circ.apply_gate('H', i, gate_round=0)

# 8 rounds of entangling gates
for r in range(1, 9):
    # even pairs
    for i in range(0, 10, 2):
        circ.apply_gate('CNOT', i, i + 1, gate_round=r)

    # Y-rotations
    for i in range(10):
        circ.apply_gate('RY', 1.234, i, gate_round=r)

    # odd pairs
    for i in range(1, 9, 2):
        circ.apply_gate('CZ', i, i + 1, gate_round=r)

# final layer of hadamards
for i in range(10):
    circ.apply_gate('H', i, gate_round=r + 1)

circ.psi.graph(color=['PSI0', 'H', 'CNOT', 'RY', 'CZ'])