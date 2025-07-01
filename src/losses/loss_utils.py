# file with the utils to compute and manage the losses

import torch

# make a permutation where every element change its position
def roll_index(features_considered):
    # pass the variable to be changed
    new_order = torch.randperm(features_considered.size()[0])
    order_indeces = torch.arange(features_considered.size()[0])
    if any(new_order == order_indeces):
        new_order = torch.roll(order_indeces, 1, dims=0)
    features_considered = features_considered[new_order]
    # return the variable changed in order to be used in the loss function
    return features_considered

def local_sup_contrastive_loss(query : torch.Tensor, positives : [torch.Tensor] , negative : torch.Tensor, loss_fn):
    loss= 0
    for positive in positives:
        loss += loss_fn(query, positive, negative)
    loss = loss / len(positives)
    return loss
