import mpi4py.MPI
import h5py
from quop_mpi import Ansatz, param
from quop_mpi.propagator import diagonal, sparse
import networkx as nx
import numpy as np
from scipy import sparse as sp

n_qubits = 5
system_size = 2 ** n_qubits
np.random.seed(1)

def quality_distribution(n_qubits):
    # return a numpy array of size "system_size"
    return np.random.normal(loc=100, scale=20, size=2 ** n_qubits)


def random_mixer(system_size, density):
    n_edges = int(np.ceil(system_size * density))
    print(f'Creating random sparse mixer with {n_edges} edges...', flush = True)
    rows = np.random.choice(n_edges, size = n_edges, replace = True)
    columns = np.random.choice(n_edges, size = n_edges, replace = True)
    values = np.ones(n_edges)
    adjacency_matrix = sp.coo_matrix((values, (rows, columns)), shape = (system_size, system_size))
    adjacency_matrix = adjacency_matrix.tocsr()
    print('...done', flush = True)
    return adjacency_matrix 


UQ = diagonal.unitary(
    diagonal.operator.serial,
    operator_kwargs={"function": quality_distribution, "args": [n_qubits]},
    parameter_function=param.rand.uniform,
)
UW = sparse.unitary(
    sparse.operator.serial,
    operator_kwargs={"function": random_mixer, "args": [system_size, 0.05]},
    parameter_function=param.rand.uniform,
)
alg = Ansatz(system_size)
alg.set_unitaries([UQ, UW])
alg.set_observables(0)
alg.set_log("sparse_random", "example", action="a")
alg.benchmark(
    range(1, 2), 1, param_persist=True, filename="sparse_random", save_action="a"
)
