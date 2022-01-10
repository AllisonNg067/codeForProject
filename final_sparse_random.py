import mpi4py.MPI
import h5py
from quop_mpi import Ansatz, param
from quop_mpi.propagator import diagonal, sparse
import networkx as nx
import numpy as np

n_qubits = 5
system_size = 2 ** n_qubits
np.random.seed(1)

def quality_distribution(n_qubits):
    # return a numpy array of size "system_size"
    return np.random.normal(loc=100, scale=20, size=2 ** n_qubits)


def random_mixer(n_qubits):
    # return a sparse adjacency matrix
    # the matrix needs to be in the 'SciPy' CSR format
    # nx.to_scipy_sparse_matrix(G, format = 'csr'
    G = nx.fast_gnp_random_graph(2 ** n_qubits, 0.3, directed=False)
    return [nx.to_scipy_sparse_matrix(G, format="csr")]


UQ = diagonal.unitary(
    diagonal.operator.serial,
    operator_kwargs={"function": quality_distribution, "args": [n_qubits]},
    parameter_function=param.rand.uniform,
)
UW = sparse.unitary(
    sparse.operator.serial,
    operator_kwargs={"function": random_mixer, "args": [n_qubits]},
    parameter_function=param.rand.uniform,
)
alg = Ansatz(system_size)
alg.set_unitaries([UQ, UW])
alg.set_observables(0)
alg.set_log("sparse_random", "example", action="a")
alg.benchmark(
    range(1, 9), 3, param_persist=True, filename="sparse_random", save_action="a"
)
