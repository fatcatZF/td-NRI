import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-layer fully connected ELU net with batch norm"""
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        #print(type(inputs))
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)




class TempMP(nn.Module):
    """Temporal Message Passing"""
    def __init__(self, n_in, n_emb, n_edge, n_node, do_prob=0.3):
        super(TempMP, self).__init__()
        self.mlp_emb = MLP(n_in, n_emb, n_emb, do_prob)
        self.mlp_e1 = MLP(2*n_emb, n_edge, n_edge)
        self.mlp_n1 = MLP(n_edge, n_node, n_node)
        self.mlp_e2 = MLP(3*n_node, n_edge, n_edge)

    def edge2node(self, x, rel_rec, rel_send):
        """
        args:
          x: [n_batch, n_edges, dim_edge]
        """
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        """
        args:
          inputs: [n_batch, n_atoms, n_in]
          rel_rec, rel_send: [n_edges, n_atoms]
        """
        x = self.mlp_emb(inputs)
        #shape: [n_batch, n_atoms, n_emb]
        x = self.node2edge(x, rel_rec, rel_send)
        #shape: [n_batch, num_edges, 2*n_emb]
        x = self.mlp_e1(x)
        #shape: [n_batch, num_edges, dim_edge]
        x_skip = x
        x = self.edge2node(x, rel_rec, rel_send)
        #shape: [n_batch, num_atoms, dim_edge]
        x = self.mlp_n1(x)
        #shape: [n_batch, num_atoms, dim_node]
        x = self.node2edge(x, rel_rec, rel_send)
        #shape: [n_batch, num_atoms, 2*dim_node]
        x = torch.cat([x, x_skip], dim=2) #skip connection
        #shape: [n_batch, num_atoms, 3*dim_node]
        x = self.mlp_e2(x)
        #shape: [n_batch, num_atoms, dim_edge]

        return x






        

