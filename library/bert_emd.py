import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel


class BERT_EMD(nn.Module):
    def __init__(self):
        super(BERT_EMD, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")

    def forward(
        self,
        q_input_ids=None,
        q_attention_mask=None,
        q_token_type_ids=None,
        d_input_ids=None,
        d_attention_mask=None,
        d_token_type_ids=None,
        reg: float = 1,
        nit: int = 100,
    ):

        # get query embeddings
        q_embeds = self.model(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
        )["last_hidden_state"]

        # get document embeddings
        d_embeds = self.model(
            input_ids=d_input_ids,
            attention_mask=d_attention_mask,
            token_type_ids=d_token_type_ids,
        )["last_hidden_state"]

        # get the cost matrix
        C = self.get_cost_matrix(q_embeds, d_embeds)

        # get query and texts distributions
        q_dist = self.get_distributions(q_attention_mask)
        d_dist = self.get_distributions(d_attention_mask)

        # solve the optimal transport problem
        T = self.sinkhorn(q_dist, d_dist, C, reg, nit)

        # calculate the distances
        distances = (C * T).view(C.shape[0], -1).sum(dim=1)
        # delete the variables that are not used
        del q_embeds, d_embeds, q_dist, d_dist

        # return the loss, transport and cost matrices
        return distances, C, T

    def get_cost_matrix(self, q_embeds: torch.Tensor, d_embeds: torch.Tensor):
        """Documentation

        """
        # normalize the embeddings
        q_embeds = f.normalize(q_embeds, p=2, dim=2)
        d_embeds = f.normalize(d_embeds, p=2, dim=2)
        # calculate and return the cost matrix
        cost_matrix = q_embeds.matmul(d_embeds.transpose(1, 2))
        cost_matrix = cost_matrix.new_ones(cost_matrix.shape) - cost_matrix
        return cost_matrix

    def get_distributions(self, attention: torch.Tensor):
        """Documentation

        """

        dist = attention.new_ones(attention.shape) * attention
        dist_sum = dist.sum(dim=1).view(-1, 1).repeat(1, attention.shape[1])
        return dist / dist_sum

    def sinkhorn(self, q_dist, d_dist, cost_matrix, reg, nit):
        """Documentation
        """

        # calculate the exponent values
        cm_max = (
            cost_matrix.amax((1, 2)).view(-1, 1, 1).repeat((1,) + cost_matrix.shape[1:])
        )
        cm = cost_matrix / cm_max / reg
        # prepare the initial variables
        K = torch.exp(-cm)
        KT = K.transpose(1, 2)
        # initialize the u tensor
        u = q_dist.new_ones(q_dist.shape)
        for i in range(nit):
            # calculate the v_{i} tensor
            v = d_dist / KT.bmm(u.view(u.shape + (1,))).squeeze()
            # calculate the u_{i} tensor
            u = q_dist / K.bmm(v.view(v.shape + (1,))).squeeze()
        # calculate the transport matrix
        U = torch.diag_embed(u)
        V = torch.diag_embed(v)
        T = U.bmm(K).bmm(V)
        # delete the variables
        del cm, cm_max, K, KT, u, v, U, V
        return T

