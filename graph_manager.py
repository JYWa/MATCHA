import collections
import networkx as nx
import random
import numpy as np
import types
import cvxpy as cp
from mpi4py import MPI



"""
GraphProcessor

:description: GraphProcessor is designed to preprocess the communication graph,
              It specifies the activated neighbors of each node at each iteration
"""

class GraphProcessor(object):
    def __init__(self, base_graph, commBudget, rank, size, iterations, issubgraph):
        self.rank = rank # index of worker
        self.size = size # totoal number of workers
        self.comm = MPI.COMM_WORLD
        self.commBudget = commBudget # user defined budget

        if issubgraph:
            # if the base graph is already decomposed
            self.base_graph = self.getGraphFromSub(base_graph)
            self.subGraphs = base_graph
        else:
            # else: decompose the base graph
            self.base_graph = base_graph
            self.subGraphs = self.getSubGraphs()

        # get Laplacian matrices for subgraphs
        self.L_matrices = self.graphToLaplacian()

        # get neighbors' index
        self.neighbors_info = self.drawer()


    def getProbability(self):
        """ compute activation probabilities for subgraphs """
        raise NotImplemented

    def getAlpha(self):
        """ compute mixing weights """
        raise NotImplemented

    def set_flags(self, iterations):
        """ generate activation flags for each iteration """
        raise NotImplemented

    def getGraphFromSub(self, subGraphs):
        G = nx.Graph()
        for edge in subGraphs:
            G.add_edges_from(edge)
        return G


    def getSubGraphs(self):
        """ Decompose the base graph into matchings """
        G = self.base_graph
        subgraphs = list()

        # first try to get as many maximal matchings as possible
        for i in range(self.size-1):
            M1 = nx.max_weight_matching(G)
            if nx.is_perfect_matching(G, M1):
                G.remove_edges_from(list(M1))
                subgraphs.append(list(M1))
            else:
                edge_list = list(G.edges)
                random.shuffle(edge_list)
                G.remove_edges_from(edge_list)
                G.add_edges_from(edge_list)
        
        # use greedy algorithm to decompose the remaining part
        rpart = self.decomposition(list(G.edges))
        for sgraph in rpart:
            subgraphs.append(sgraph)

        return subgraphs

    def graphToLaplacian(self):
        L_matrices = list()
        for i, subgraph in enumerate(self.subGraphs):
            tmp_G = nx.Graph()
            tmp_G.add_edges_from(subgraph)
            L_matrices.append(nx.laplacian_matrix(tmp_G, list(range(self.size))).todense())

        return L_matrices


    def decomposition(self, graph):
        size = self.size

        node_degree = [[i, 0] for i in range(size)]
        node_to_node = [[] for i in range(size)]
        node_degree_dict = collections.defaultdict(int)
        node_set = set()
        for edge in graph:
            node1, node2 = edge[0], edge[1]
            node_degree[node1][1] += 1
            node_degree[node2][1] += 1
            if node1 in node_to_node[node2] or node2 in node_to_node[node1]:
                print("Invalid input graph! Double edge! ("+str(node1) +", "+ str(node2)+")")
                exit()
            if node1 == node2:
                print("Invalid input graph! Circle! ("+str(node1) +", "+ str(node2)+")")
                exit()
 
            node_to_node[node1].append(node2)
            node_to_node[node2].append(node1)
            node_degree_dict[node1] += 1
            node_degree_dict[node2] += 1
            node_set.add(node1)
            node_set.add(node2)

        node_degree = sorted(node_degree, key = lambda x: x[1])
        node_degree[:] = node_degree[::-1]
        subgraphs = []
        min_num = node_degree[0][1]
        while node_set:
            subgraph = []
            for i in range(size):
                node1, node1_degree = node_degree[i]
                if node1 not in node_set:
                    continue
                for j in range(i+1, size):
                    node2, node2_degree = node_degree[j]
                    if node2 in node_set and node2 in node_to_node[node1]:
                        subgraph.append((node1, node2))
                        node_degree[j][1] -= 1
                        node_degree[i][1] -= 1
                        node_degree_dict[node1] -= 1
                        node_degree_dict[node2] -= 1
                        node_to_node[node1].remove(node2)
                        node_to_node[node2].remove(node1)
                        node_set.remove(node1)
                        node_set.remove(node2)
                        break
            subgraphs.append(subgraph)
            for node in node_degree_dict:
                if node_degree_dict[node] > 0:
                    node_set.add(node)
            node_degree = sorted(node_degree, key = lambda x: x[1])
            node_degree[:] = node_degree[::-1]
        return subgraphs

    def drawer(self):
        """
        input graph: list[list[tuples]]
                     [graph1, graph2,...]
                     graph: [edge1, edge2, ...]
                     edge: (node1, node2)
        output connect: matrix: [[]]
        """
        
        connect = []
        cnt = 1
        for graph in self.subGraphs:
            new_connect = [-1 for i in range(self.size)]
            for edge in graph:
                node1, node2 = edge[0], edge[1]
                if new_connect[node1] != -1 or new_connect[node2] != -1:
                    print("invalide graph! graph: "+str(cnt))
                    exit()
                new_connect[node1] = node2
                new_connect[node2] = node1
            # print(new_connect)
            connect.append(new_connect)
            cnt += 1
        return connect

class FixedProcessor(GraphProcessor):
    """ wrapper for fixed communication graph """

    def __init__(self, base_graph, commBudget, rank, size, iterations, issubgraph):
        super(FixedProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, issubgraph)
        self.probabilities = self.getProbability()
        self.neighbor_weight = self.getAlpha()
        self.active_flags = self.set_flags(iterations + 1)


    def getProbability(self):
        """ activation probabilities are same for subgraphs """
        return self.commBudget 

    def getAlpha(self):
        """ there is an analytical expression of alpha in this case """
        L_base = np.zeros((self.size, self.size))
        for subLMatrix in self.L_matrices:
            L_base += subLMatrix
        w_b, _ = np.linalg.eig(L_base)
        lambdaList = list(sorted(w_b))
        if len(w_b) > 1:
            alpha = 2 / (lambdaList[1] + lambdaList[-1])

        return alpha

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers 
                     so that the activation flags are same

        """
        iterProb = np.random.binomial(1, self.probabilities, iterations)
        flags = list()
        idx = 0
        for prob in iterProb:
            if idx % 2 == 0:
                flags.append([0,1])
            else:
                flags.append([1,0])
            
            idx += 1
            # flags.append([prob for i in range(len(self.L_matrices))])

        return flags

class MatchaProcessor(GraphProcessor):
    """ Wrapper for MATCHA
        At each iteration, only a random subset of subgraphs are activated
    """

    def __init__(self, base_graph, commBudget, rank, size, iterations, issubgraph):
        super(MatchaProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, issubgraph)
        self.probabilities = self.getProbability()
        self.neighbor_weight = self.getAlpha()
        self.active_flags = self.set_flags(iterations + 1)



    def getProbability(self):
        num_subgraphs = len(self.L_matrices)
        p = cp.Variable(num_subgraphs)
        L = p[0]*self.L_matrices[0]
        for i in range(num_subgraphs-1):
            L += p[i+1]*self.L_matrices[i+1]
        eig = cp.lambda_sum_smallest(L, 2)
        sum_p = p[0]
        for i in range(num_subgraphs-1):
            sum_p += p[i+1]

        # cvx optimization for activation probabilities
        obj_fn = eig
        constraint = [sum_p <= num_subgraphs*self.commBudget, p>=0, p<=1]
        problem = cp.Problem(cp.Maximize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        # get solution
        tmp_p = p.value
        originActivationRatio = np.zeros((num_subgraphs))
        for i, pval in enumerate(tmp_p):
            originActivationRatio[i] = np.real(float(pval))

        return np.minimum(originActivationRatio, 1) 

    def getAlpha(self):
        num_subgraphs = len(self.L_matrices)
        num_nodes = self.size

        # prepare matrices
        I = np.eye(num_nodes)
        J = np.ones((num_nodes, num_nodes))/num_nodes
        
        mean_L = np.zeros((num_nodes,num_nodes))
        var_L = np.zeros((num_nodes,num_nodes))
        for i in range(num_subgraphs):
            val = self.probabilities[i]
            mean_L += self.L_matrices[i]*val
            var_L += self.L_matrices[i]*(1-val)*val
        
        # SDP for mixing weight
        a = cp.Variable()
        b = cp.Variable()
        s = cp.Variable()
        obj_fn = s
        constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
        problem = cp.Problem(cp.Minimize(obj_fn), constraint)
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        return  float(a.value)


    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers 
                     so that the activation flags are same

        """
        flags = list()
        for i in range(len(self.L_matrices)):
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))


        return [list(x) for x in zip(*flags)]