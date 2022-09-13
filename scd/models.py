#
# SPDX-FileCopyrightText: 2022 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from Uncertainty_aware_SSL.utils.losses import STDMinimizer, STDCapRegularizer


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class BatchNormWrapper(nn.Module):
    def __init__(self, m):
        super(BatchNormWrapper, self).__init__()
        self.m = m
        self.m.eval()  # Set the batch norm to eval mode

    def forward(self, x):
        input_type = x.dtype
        x = self.m(x.float())
        return x.to(input_type)


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)

    # SCD - BEGIN: projector
    sizes = [cls.config.embedding_dim] + \
            list(map(int, cls.config.projector.split('-')))
    projectors, normalizers = [], []
    for _ in range(cls.model_args.n_projectors):
        layers = []
        for j in range(len(sizes) - 2):
            if j == 0:
                layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=True))
            else:
                layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[j + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        projectors.append(nn.Sequential(*layers))
    normalizers = [
        nn.BatchNorm1d(sizes[-1], affine=False)
        for _ in range(cls.model_args.n_projectors)
    ]
    cls.projector = nn.ModuleList(projectors)
    cls.bn = nn.ModuleList(normalizers)
    # cls.proj_bn = nn.ModuleList([bn(proj) for proj, bn in zip(projectors, normalizers)])
    # SCD - END: projector

    cls.init_weights()


def normalize(vec):
    return vec.div(torch.norm(vec, dim=-1).unsqueeze(-1))


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cl_forward(cls,
               encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None,
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None,
               mlm_input_ids=None,
               mlm_labels=None,
               ):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view(
        (-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in [
            'avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in [
                'avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view(
        (batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # SCD BEGIN: SimCSE legacy code

    # Hard negative
    # if num_sent == 3:
    #    z3 = pooler_output[:, 2]

    # SCD END: SimCSE legacy code

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        # if num_sent >= 3:
        #     z3_list = [torch.zeros_like(z3)
        #                for _ in range(dist.get_world_size())]
        #     dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
        #     z3_list[dist.get_rank()] = z3
        #     z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    lambd = cls.config.task_lambda

    # empirical cross-correlation matrix
    # LISA add multiple heads
    res1, res2 = [], []
    cross_corr = []
    for i in range(cls.model_args.n_projectors):
        proj1 = cls.projector[i](z1)
        bn1 = cls.bn[i](proj1)
        res1.append(bn1)
        proj2 = cls.projector[i](z2)
        bn2 = cls.bn[i](proj2)
        res2.append(bn2)
        cross_corr.append(bn1.T @ bn2)

    feat1 = torch.mean(torch.stack(res1), dim=0)
    feat2 = torch.mean(torch.stack(res2), dim=0)
    if cls.model_args.n_projectors > 1:
        feat1_std = torch.sqrt(torch.var(torch.stack(res1), dim=0) + 0.0001)
        feat2_std = torch.sqrt(torch.var(torch.stack(res2), dim=0) + 0.0001)
    else:  # otherwise, torch.var produces nan
        feat1_std = torch.zeros(feat1.shape)
        feat2_std = torch.zeros(feat2.shape)
    features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
    features_std = torch.cat([feat1_std.unsqueeze(1), feat2_std.unsqueeze(1)], dim=1)

    c = sum(cross_corr) / len(cross_corr)

    # sum the cross-correlation matrix between all gpus
    c.div_(len(z1))

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    decorrelation = on_diag + lambd * off_diag

    self_contrast = torch.diag(cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))).mean()

    loss = (
        cls.config.task_alpha * self_contrast + cls.config.task_beta * decorrelation
    )

    # SCD BEGIN: SimCSE legacy code

    # # Hard negative
    # if num_sent >= 3:
    #     z1_z3_cos = cls.sim(z1.unsqueeze(
    #         1), (z3+eps*dec_grad3).unsqueeze(0))
    #     z1_z3_cos[range(len(z1_z3_cos)), range(len(z1_z3_cos))] = 0
    #     cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    # loss_fct = nn.CrossEntropyLoss()

    # # Calculate loss with hard negatives
    # if num_sent == 3:
    #     # Note that weights are actually logits of weights
    #     z3_weight = cls.model_args.hard_negative_weight
    #     weights = torch.tensor(
    #         [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [
    #             0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(cls.device)
    #     cos_sim = cos_sim + weights

    # loss = loss_fct(cos_sim, labels)

    # SCD END: SimCSE legacy code

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss += cls.model_args.mlm_weight * masked_lm_loss

    # LISA add loss component
    mean_std = torch.sum(features_std) / features.shape[0]
    print(f'---> feature STD: {mean_std:.2f}')

    if cls.model_args.lambda2_unc > 0:
        uncertainty_regularizer = STDCapRegularizer(
            features, features_std, cls.model_args.lambda2_unc
        )
        uncertainty_penalty = uncertainty_regularizer.loss()
    else:
        uncertainty_regularizer = STDMinimizer(features, features_std)
        uncertainty_penalty = uncertainty_regularizer.loss()

    if torch.cuda.is_available():
        uncertainty_penalty = uncertainty_penalty.cuda()
    print(f'---> loss SCD: {loss:.2f}, loss OURS: {uncertainty_penalty:.2f}')
    loss += cls.model_args.lambda1_unc * uncertainty_penalty
    print(f'---> total loss: {loss:.2f}')

    if not return_dict:
        output = (loss,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=loss,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in [
            'avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    # SCD BEGIN: SimCSE legacy code
    # if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
    #     pooler_output = cls.mlp(pooler_output)
    # SCD END: SimCSE legacy code

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,
                                   )
        else:
            return cl_forward(self, self.bert,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              mlm_input_ids=mlm_input_ids,
                              mlm_labels=mlm_labels,
                              )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,
                                   )
        else:
            return cl_forward(self, self.roberta,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              mlm_input_ids=mlm_input_ids,
                              mlm_labels=mlm_labels,
                              )
