import networkx as nx
import matplotlib.pyplot as plt
from .model import create_model

def visualize_network():
    model = create_model()
    edges = model.edges()

    G = nx.DiGraph()
    G.add_edges_from(edges)

    plt.figure()
    nx.draw(G, with_labels=True)
    plt.title("Bayesian Network Structure for Medical Diagnosis")
    plt.show()

if __name__ == "__main__":
    visualize_network()