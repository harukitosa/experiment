# てンソルネットワークを用いて内積計算を行う
import tensornetwork as tn
import numpy as np
import math
# import jax

tn.set_default_backend("jax")

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

class ValidationError(Exception):
    pass 

def RX(theta):
	return np.array([[math.cos(theta/2), -math.sin(theta/2)*1.0j], [-math.sin(theta/2)*1.0j, math.cos(theta/2)]], dtype=complex)

def calc_inner_product(theta_x, theta_y, qubit_size):
  if len(theta_x) != qubit_size:
    raise ValidationError("theta_xのsizeが:{}でqubits数:{}と異なります".format(len(theta_x), qubit_size))
  if len(theta_y) != qubit_size:
    raise ValidationError("theta_yのsizeが{}でqubits数と異なります".format(len(theta_y), qubit_size))

  all_nodes = []
  with tn.NodeCollection(all_nodes):
    state_nodes = [
        tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j],)) for _ in range(qubit_size)
    ]
    qubits = [node[0] for node in state_nodes]
    for theta in theta_x:
      apply_gate(qubits, RX(math.pi*theta), [0]) 

    apply_gate(qubits, CNOT, [0, 1])
    apply_gate(qubits, CNOT, [2, 3])
    apply_gate(qubits, CNOT, [2, 3])
    apply_gate(qubits, CNOT, [0, 1])

    for theta in theta_y:
      apply_gate(qubits, RX(math.pi*theta), [0]) 
    return qubits, all_nodes

result = calc_inner_product([0.1, 0.2, 0.3, 0.4], [0.1, 0.3, 0.123, 0.4], 4)

result = tn.contractors.optimal(
    result[1], output_edge_order=result[0])


print(np.abs(result.tensor[0][0][0][0]))
