import mpi4py.MPI
import h5py
from quop_mpi.algorithm import qwoa
from quop_mpi import observable
from quop_mpi.toolkit import I, Z
import networkx as nx
from numpy import random
Graph = nx.circular_ladder_graph(4)

vertices = len(Graph.nodes)
system_size = 2 ** vertices

G = nx.to_scipy_sparse_matrix(Graph)
random.seed(1)
#Modify it 
def maxcut_qualities(n_qubits):
    array_length = 2**n_qubits
    qs = random.normal(loc = 100, scale = 20, size=array_length)
    print(qs)
    return qs


alg = qwoa(system_size)
alg.set_qualities(observable.serial, {"function": maxcut_qualities, "args": [vertices]})
alg.set_log("final_qwoa_maxcut.csv", "final_qwoa_maxcut", action="w")
alg.set_depth(2)
alg.benchmark(range(1, 20), 3, param_persist=True, filename="qwoa_results", save_action="w")
