import tensornetwork as tn
import numpy as np

state = tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j]))

# https://github.com/google/TensorNetwork/blob/master/docs/quantum_circuit.rst
def apply_gate(qubit_edges, gate, operating_qubits):
  op = tn.Node(gate)
  for i, bit in enumerate(operating_qubits):
    tn.connect(qubit_edges[bit], op[i])
    qubit_edges[bit] = op[i + len(operating_qubits)]

# These are just numpy arrays of the operators.
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.zeros((2, 2, 2, 2), dtype=complex)
CNOT[0][0][0][0] = 1
CNOT[0][1][0][1] = 1
CNOT[1][0][1][1] = 1
CNOT[1][1][1][0] = 1
all_nodes = []
# NodeCollection allows us to store all of the nodes created under this context.
with tn.NodeCollection(all_nodes):
  # state_nodes = [
  #     tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j],)) for _ in range(2)
  # ]
  state_nodes = [
    tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j])),
    tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j])),
  ]
  print(len(state_nodes))
  print(state_nodes[0].tensor)
  print(state_nodes[1].tensor)

  qubits = [node[0] for node in state_nodes]
  apply_gate(qubits, H, [0])
  apply_gate(qubits, CNOT, [0, 1])
# We can contract the entire tensornetwork easily with a contractor algorithm
result = tn.contractors.optimal(
    all_nodes, output_edge_order=qubits)

print(result.tensor[0][0]) # array([0.707+0.j, 0.0+0.j], [0.0+0.j, 0.707+0.j])
print(result.tensor[1][1]) # array([0.707+0.j, 0.0+0.j], [0.0+0.j, 0.707+0.j])
print(result.tensor[1][0]) # array([0.707+0.j, 0.0+0.j], [0.0+0.j, 0.707+0.j])
print(result.tensor[0][1]) # array([0.707+0.j, 0.0+0.j], [0.0+0.j, 0.707+0.j])
