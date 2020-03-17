
# MATCHA: Communication-Efficient Decentralized SGD

Code to reproduce the experiments reported in this paper:
> Jianyu Wang, Anit Kumar Sahu, Zhouyi Yang, Gauri Joshi, Soummya Kar, "[MATCHA: Speeding Up Decentralized SGD via Matching Decomposition Sampling](https://arxiv.org/abs/1905.09435)," arxiv preprint 2019.

A short version has been abridged in [FL-NeurIPS'19](http://federated-learning.org/fl-neurips-2019/) and received the Distinguished Student Paper Award.

This repo contains the implementations of MATCHA and [D-PSGD](https://papers.nips.cc/paper/7117-can-decentralized-algorithms-outperform-centralized-algorithms-a-case-study-for-decentralized-parallel-stochastic-gradient-descent.pdf) for any arbitrary node topologies. You can also use it to develop other decentralized training methods. Please cite this paper if you use this code for your research/projects.

## Dependencies and Setup
The code runs on Python 3.5 with PyTorch 1.0.0 and torchvision 0.2.1.
The peer-to-peer communication among workers is achieved by [MPI4Py](https://mpi4py.readthedocs.io/en/stable/) sendrecv function.

## Training examples
Here is an example on how to use MATCHA to train a neural network.
```python
import util
from graph_manager import FixedProcessor, MatchaProcessor
from communicator import decenCommunicator, ChocoCommunicator, centralizedCommunicator

# Define the base node topology by giving the graph ID
# There are six pre-defined graphs in utils.py
base_graph = util.select_graph(args.graphid)

# Preprocess the base topology: 1) decompose it into matchings; 
#                               2) get activation probabities for matchings;
#                               3) compute the mixing weight;
#                               4) generate activation flags for each iteration
# All these information is stored in GP
GP = MatchaProcessor(base_graph, 
                     commBudget = args.budget,
                     rank = rank,
                     size = size,
                     iterations = args.epoch * num_batches,
                     issubgraph = True)

# Define the communicator
communicator = decenCommunicator(rank, size, GP)

# Start training
for batch_id, (data, label) in enumerate(data_loader):
    # same as serial training
    output = model(data) # forward
    loss = criterion(output, label)
    loss.backward() # backward
    optimizer.step() # gradient step
    optimizer.zero_grad()

    # additional line to average local models at workers
    communicator.communicate(model)
```
In order to use [D-PSGD](https://papers.nips.cc/paper/7117-can-decentralized-algorithms-outperform-centralized-algorithms-a-case-study-for-decentralized-parallel-stochastic-gradient-descent.pdf), we just need to change `MatchaProcessor` to `FixedProcessor`. Similarly, in order to use [ChocoSGD](https://arxiv.org/abs/1902.00340), we can change `MatchaProcessor` to `FixedProcessor` and `decenCommunicator` to `ChocoCommunicator`.  If one wants to run fully synchronous SGD, then `centralizedCommunicator` can be used and there is no need to define the graph processor.

In addition, before training starts, we need to initialize MPI processes on each worker machine as follows:
```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```
The script can be run using the following command:
```shell
mpirun --hostfile c8 -np 8 python train_mpi.py
```

## Citation
```
@article{wang2019matcha,
  title={{MATCHA}: Speeding Up Decentralized {SGD} via Matching Decomposition Sampling},
  author={Wang, Jianyu and Sahu, Anit Kumar and Yang, Zhouyi and Joshi, Gauri and Kar, Soummya},
  journal={arXiv preprint arXiv:1905.09435},
  year={2019}
}
```
