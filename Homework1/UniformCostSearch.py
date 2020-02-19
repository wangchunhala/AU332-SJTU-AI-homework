# -*- coding: utf-8 -*-
from queue import LifoQueue
from queue import Queue
from queue import PriorityQueue

class Graph:
    """
    Defines a graph with edges, each edge is treated as dictionary
    look up. function neighbors pass in an id and returns a list of 
    neighboring node
    
    """
    def __init__(self):
        self.edges = {}
        self.edgeWeights = {}
        self.locations = {}

    def neighbors(self, id):
        if id in self.edges:
            return self.edges[id]
        else:
            print("The node ", id , " is not in the graph")
            return False

    def get_node_location(self, id):
        return self.nodeLocation[id]

    def get_cost(self,from_node, to_node):
        #print("get_cost for ", from_node, to_node)
        nodeList = self.edges[from_node]
        #print(nodeList)
        try:
            edgeList = self.edgeWeights[from_node]
            return edgeList[nodeList.index(to_node)]
        except ValueError:
            print("From node ", from_node, " to ", to_node, " does not exist a direct connection")
            return False

def reconstruct_path(came_from, start, goal):
    """
    Given a dictionary of came_from where its key is the node 
    character and its value is the parent node, the start node
    and the goal node, compute the path from start to the end

    Arguments:
    came_from -- a dictionary indicating for each node as the key and 
                 value is its parent node
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    path. -- A list storing the path from start to goal. Please check 
             the order of the path should from the start node to the 
             goal node
    """
    path = []
    ### START CODE HERE ### (≈ 6 line of code)
    if(came_from == {}):
        return {}
    else:
     while(1):
        if (goal == start):
            break
        else:
                path.append(goal)
                goal = came_from[goal]
    path.append(start)

    path.reverse()




    ### END CODE HERE ###
    return path


def uniform_cost_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize uniform cost search algorithm by finding the path from 
    start node to the goal node
    Use early stoping in your code
    This function returns back a dictionary storing the information of each node
    and its corresponding parent node
    Arguments:
    graph -- A dictionary storing the edge information from one node to a list 
             of other nodes
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    came_from -- a dictionary indicating for each node as the key and 
                value is its parent node
    """
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    ### START CODE HERE ### (≈ 15 line of code)
    if start not in graph.edges:
          print(" the ", start, "not exist in",graph)
          return {},{}
    elif goal not in graph.edges:
         print(" the ", goal, "not exist in",graph)
         return {},{}

    else:
       sq=PriorityQueue()
       visited=[]
       sq._put((0,start))

       while sq:

        parent=sq._get()[1]
        target = graph.edges[parent]
        targetcost=graph.edgeWeights[parent]
        if parent not in visited:
            visited.append(parent)
            if parent == goal:
             return came_from, cost_so_far
            for i in target:
                if (i not in cost_so_far) or (cost_so_far[i] > targetcost[target.index(i)] + cost_so_far[parent]):
                     came_from[i]=parent
                     print(came_from)

                     cost_so_far[i]=(targetcost[target.index(i)]+cost_so_far[parent])
                     sq._put((cost_so_far[i],i))
                     print(sq.queue)


    # target = graph.edges[start]
    # targetcost=graph.edgeWeights[start]
    # parent = start
    # while goal not in target:


    #     for item in target:
    #         if (item not in cost_so_far) or (cost_so_far[item]>targetcost[target.index(item)]+cost_so_far[parent]):
    #             came_from[item] = parent
    #             cost_so_far[item]=targetcost[target.index(item)]+cost_so_far[parent]
    #             sq._put((cost_so_far[item], item))
    #     parent= sq._get()[1]
    #     targetcost=graph.edgeWeights[parent]
    #     target = graph.edges[parent]
    # came_from[goal] = parent
    # cost_so_far[goal]=targetcost[target.index(goal)]+cost_so_far[parent]
    ### END CODE HERE ###
    return came_from, cost_so_far




# The main function will first create the graph, then use uniform cost search
# which will return the came_from dictionary 
# then use the reconstruct path function to rebuild the path.
if __name__=="__main__":
    small_graph = Graph()
    small_graph.edges = {
        'A': ['B','D','C'],
        'B': [ 'D'],
        'C': ['B','E'],
        'D': ['E'],
        'E': ['B']
    }
    small_graph.edgeWeights={
        'A': [10,20,3],
        'B': [5],
        'C': [2,15],
        'D': [11],
        'E': [5]
    }

    # large_graph = Graph()
    # large_graph.edges = {
    #     'S': ['A','B','C'],
    #     'A': ['S','B','D'],
    #     'B': ['S', 'A', 'D','H'],
    #     'C': ['S','L'],
    #     'D': ['A', 'B','F'],
    #     'E': ['G','K'],
    #     'F': ['H','D'],
    #     'G': ['H','E'],
    #     'H': ['B','F','G'],
    #     'I': ['L','J','K'],
    #     'J': ['L','I','K'],
    #     'K': ['I','J','E'],
    #     'L': ['C','I','J']
    # }
    # large_graph.edgeWeights = {
    #     'S': [7, 2, 3],
    #     'A': [7, 3, 4],
    #     'B': [2, 3, 4, 1],
    #     'C': [3, 2],
    #     'D': [4, 4, 5],
    #     'E': [2, 5],
    #     'F': [3, 5],
    #     'G': [2, 2],
    #     'H': [1, 3, 2],
    #     'I': [4, 6, 4],
    #     'J': [4, 6, 4],
    #     'K': [4, 4, 5],
    #     'L': [2, 4, 4]
    # }

    print("Small graph")
    start = 'r'
    goal = 'E'
    came_from_UCS, cost_so_far = uniform_cost_search(small_graph, start, goal)
    print("came from UCS " , came_from_UCS)
    print("cost form UCS ", cost_so_far)
    pathUCS = reconstruct_path(came_from_UCS, start, goal)
    print("path from UCS ", pathUCS)

    # print("Large graph")
    # start = 'S'
    # goal = 'E'
    # came_from_UCS, cost_so_far = uniform_cost_search(large_graph, start, goal)
    # print("came from UCS " , came_from_UCS)
    # print("cost form UCS ", cost_so_far)
    # pathUCS = reconstruct_path(came_from_UCS, start, goal)
    # print("path from UCS ", pathUCS)