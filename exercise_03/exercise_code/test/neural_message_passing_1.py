from .base_tests import UnitTest, CompositeTest, test_results_to_score
from exercise_code.model.assign_net import BipartiteNeuralMessagePassingLayer
import torch

class Test_Shape_EdgeUpdate(UnitTest):
    def __init__(self):
        self.edge_dim = 9
        self.node_dim = 39
        self.mpn_layer = BipartiteNeuralMessagePassingLayer(self.node_dim, self.edge_dim)
        self.edge_embeds = torch.rand(20, 11, self.edge_dim *2)
        self.nodes_a_embeds = torch.rand(20, self.node_dim)
        self.nodes_b_embeds = torch.rand(11, self.node_dim)
        
    def test(self):
        edge = self.mpn_layer.edge_update(self.edge_embeds, self.nodes_a_embeds, self.nodes_b_embeds)
        return edge.shape==(20, 11, self.edge_dim)
        
    def define_success_message(self):
        return f"Congratulations: The shape of the Edge Update seems to be correct"
    def define_failure_message(self):
        return f"The shape of the Edge Update does not seem to be correct"

class Test_Shape_NodeUpdate(UnitTest):
    def __init__(self):
        self.edge_dim = 9
        self.node_dim = 39
        self.mpn_layer = BipartiteNeuralMessagePassingLayer(self.node_dim, self.edge_dim)
        self.edge_embeds = torch.rand(20, 11, self.edge_dim)
        self.nodes_a_embeds = torch.rand(20, self.node_dim)
        self.nodes_b_embeds = torch.rand(11, self.node_dim)
        
    def test(self):
        nodes_a, nodes_b = self.mpn_layer.node_update(self.edge_embeds, self.nodes_a_embeds, self.nodes_b_embeds)
        return nodes_a.shape==(20, self.node_dim) and nodes_b.shape==(11, self.node_dim)
        
    def define_success_message(self):
        return f"Congratulations: The shape of the Node Update seems to be correct"
    def define_failure_message(self):
        return f"The shape of the Node Update does not seem to be correct"

class Test_EdgeMLP_in(UnitTest):
    def __init__(self):
        self.edge_dim = 9
        self.node_dim = 39
        self.mpn_layer = BipartiteNeuralMessagePassingLayer(self.node_dim, self.edge_dim)

        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        self.edge_embeds = torch.load("exercise_code/test/neural_message_passing/nmp_1_edge_double.pth")
        self.nodes_a_embeds = torch.load("exercise_code/test/neural_message_passing/nmp_1_node_a.pth")
        self.nodes_b_embeds = torch.load("exercise_code/test/neural_message_passing/nmp_1_node_b.pth")
        
    def test(self):
        self.mpn_layer.edge_update(self.edge_embeds, self.nodes_a_embeds, self.nodes_b_embeds)
        self.nmp_1_edge_in = torch.load("exercise_code/test/neural_message_passing/nmp_1_edge_in.pth")
        return torch.allclose(self.mpn_layer.edge_in, self.nmp_1_edge_in)
        
    def define_success_message(self):
        return f"Congratulations: The input to the edge MLP seems to be correct"
    def define_failure_message(self):
        return f"The input to the edge MLP does not seem to be correct"

class Test_NodeMLP_in(UnitTest):
    def __init__(self):
        self.edge_dim = 9
        self.node_dim = 39
        self.mpn_layer = BipartiteNeuralMessagePassingLayer(self.node_dim, self.edge_dim)

        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        self.edge_embeds = torch.load("exercise_code/test/neural_message_passing/nmp_1_edge.pth")
        self.nodes_a_embeds = torch.load("exercise_code/test/neural_message_passing/nmp_1_node_a.pth")
        self.nodes_b_embeds = torch.load("exercise_code/test/neural_message_passing/nmp_1_node_b.pth")
        
    def test(self):
        self.mpn_layer.node_update(self.edge_embeds, self.nodes_a_embeds, self.nodes_b_embeds)
        self.nodes_a_in = torch.load("exercise_code/test/neural_message_passing/nmp_1_node_a_in.pth")
        self.nodes_b_in = torch.load("exercise_code/test/neural_message_passing/nmp_1_node_b_in.pth")
        return torch.allclose(self.mpn_layer.nodes_a_in, self.nodes_a_in) and torch.allclose(self.mpn_layer.nodes_b_in, self.nodes_b_in) 
        
    def define_success_message(self):
        return f"Congratulations: The input to the node MLP seems to be correct"
    def define_failure_message(self):
        return f"The input to the node MLP does not seem to be correct"


class NeuralMessagePassing(CompositeTest):
    def define_tests(self):
        pass
        return [
            Test_Shape_EdgeUpdate(),
            Test_Shape_NodeUpdate(),
            Test_NodeMLP_in(),
            Test_EdgeMLP_in()
        ]

    def define_method_name(self):
        return "neural_message_passing_1"

def test_neural_message_passing_1():
    test = NeuralMessagePassing()
    return test_results_to_score(test())