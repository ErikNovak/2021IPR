import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel


class BERT_MAX(nn.Module):
    def __init__(self):
        super(BERT_MAX, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")

    def forward(
        self,
        q_input_ids=None,
        q_attention_mask=None,
        q_token_type_ids=None,
        d_input_ids=None,
        d_attention_mask=None,
        d_token_type_ids=None,
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

        # normalize the embeddings and average them
        q_embeds = q_embeds * q_attention_mask.unsqueeze(2).repeat(
            1, 1, q_embeds.shape[2]
        )
        d_embeds = d_embeds * d_attention_mask.unsqueeze(2).repeat(
            1, 1, d_embeds.shape[2]
        )
        q_embeds, _ = q_embeds.max(dim=1)
        d_embeds, _ = d_embeds.max(dim=1)

        q_embeds = f.normalize(q_embeds, p=2, dim=1)
        d_embeds = f.normalize(d_embeds, p=2, dim=1)

        # calculate the mean distances
        distances = q_embeds.matmul(d_embeds.T)
        distances = distances.new_ones(distances.shape) - distances
        # calculate the loss value
        return distances

