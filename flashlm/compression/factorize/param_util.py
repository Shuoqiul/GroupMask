import torch
import torch.nn as nn
import torch.nn.functional as F
# from misc_functions import custom_grad_weight
import numpy as np
import math

import transformers
# from .hypernetwork import virtual_operation, hypernetwork
# from transformers.trainer_rank_pruning import WSVD
from .factorization_helper import WSVD

class collect_info_reg(nn.Module):
    def __init__(self, model, p, lam=4.0, width_flag=False):
        super(collect_info_reg, self).__init__()
        self.sum_ori_params = 0
        self.sum_svd_params = 0
        self.p = p
        self.in_dim_list = []
        self.out_dim_list = []
        self.structures = []
        self.lam = lam
        self.a_lam = 6.0
        self.width_flag = width_flag

        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if isinstance(m, WSVD):
                ori_param, svd_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.sum_svd_params += svd_param
                self.in_dim_list.append(m.ori_in_dim)
                self.out_dim_list.append(m.ori_out_dim)
                self.structures.append(m.mid_dim)
            # elif isinstance(m, transformers.trainer_pruning.Truncate_Linear):
            #     ori_param, svd_param = m.get_parameters()
            #     self.sum_ori_params += ori_param
            #     self.sum_svd_params += svd_param
            #     self.in_dim_list.append(m.ori_in_dim)
            #     self.out_dim_list.append(m.ori_out_dim)
            #     self.structures.append(m.ori_in_dim)

        # print(self.sum_ori_params)
        # print(self.sum_svd_params)
        print("number of oringal parameters: %.3f" % (self.sum_ori_params / 10 ** 6))
        print("number of svd parameters: %.3f" % (self.sum_svd_params / 10 ** 6))

    def count_current_params(self, vectors):
        with torch.no_grad():
            sum_params = 0
            for i in range(len(self.structures)):
                if self.width_flag:
                    mid_dim = vectors[i]
                else:
                    mid_dim = vectors[i].sum()
                current_params = mid_dim * self.in_dim_list[i] + mid_dim * self.out_dim_list[i]
                sum_params += current_params
        print("current parameters: %.3f" % (sum_params / 10 ** 6))
        return sum_params

    def forward(self, vectors):
        sum_params = 0
        for i in range(len(self.structures)):
            if self.width_flag:
                mid_dim = vectors[i]
            else:
                mid_dim = vectors[i].sum()
            current_params = mid_dim*self.in_dim_list[i] + mid_dim*self.out_dim_list[i]
            sum_params+=current_params

        param_ratio = sum_params / (self.sum_ori_params)

        # if self.width_flag:
        #     loss = torch.log(1+(self.p-param_ratio).abs())
        #
        # else:

        if param_ratio>self.p:

            clampled_p_ratio = torch.clamp(param_ratio, min=self.p)

            loss = torch.log(clampled_p_ratio/self.p)
        else:
            clampled_p_ratio = torch.clamp(param_ratio, max=self.p)

            loss = torch.log(self.p/clampled_p_ratio)

        return self.lam * loss

    def extra_alignment(self, soft_vectors, hard_vectors):
        sum_loss = 0
        for i in range(len(self.structures)):
            width = hard_vectors[i].sum().detach().item()
            target_vector = torch.zeros(hard_vectors[i].size(0))
            target_vector[:int(width)] = 1
            if hard_vectors[i].get_device() == -1:
                target_vector = target_vector.cpu()
            else:
                target_vector = target_vector.to(hard_vectors[i].get_device())
            current_loss = torch.nn.BCELoss()(soft_vectors[i],target_vector)
            # current_loss = torch.nn.functional.mse_loss(soft_vectors[i],target_vector)
            sum_loss+=current_loss
        loss = sum_loss/len(self.structures)
        return self.a_lam*loss

class help_functions_hn(nn.Module):
    def __init__(self, structures, width_flag=False):
        self.structures = structures
        self.width_flag = width_flag

    def print_info(self,vectors):
        print(self.structures)
        config = []
        for i in range(len(vectors)):
            config.append(vectors[i].sum().item())

        print(config)

    def set_gate_vectors(self, model, vectors):
        modules = list(model.modules())
        ind = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if isinstance(m, WSVD):
                current_vector = vectors[ind]
                m.set_vector_value(current_vector)
                ind+=1
            # elif isinstance(m, transformers.trainer_pruning.Truncate_Linear):
            #     current_vector = vectors[ind]
            #     m.set_vector_value(current_vector)
            #     ind+=1

    def set_imp_vectors(self, model, imp_vectors):
        modules = list(model.modules())
        ind = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if isinstance(m, WSVD):
                current_vector = imp_vectors[ind]
                m.set_importnace_score(current_vector)
                ind += 1

    # def collect_Q_from_model(self, model):
    #     modules = list(model.modules())
    #     ind = 0
    #     Q_list = []
    #     for layer_id in range(len(modules)):
    #         m = modules[layer_id]
    #         if isinstance(m, transformers.trainer_pruning.Truncate_Linear):
    #             Q_list.append(m.get_importance_scores())
    #             ind += 1
    #     return Q_list


    def freeze_model_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def active_model_grad(self, model):
        for param in model.parameters():
            param.requires_grad = True


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    if hasattr(model, "_orig_mod"):
        return unwrap_model(model._orig_mod)
    elif hasattr(model, "_fsdp_wrapped_module"):
        return unwrap_model(model._fsdp_wrapped_module)
    else:
        return model
