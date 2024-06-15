import random
import numpy as np
import scipy.spatial as sp
from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.xent_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.temp = temp

    def supervised_CL_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        exp_logits = torch.exp(logits)
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def cosine_similarity(self, tensor1, tensor2):
      if tensor1.dim() != tensor2.dim():
        raise ValueError("The number of dimensions are not the same.")
      dot_product = torch.dot(tensor1, tensor2.view(*tensor1.size()))

      norm1 = torch.linalg.norm(tensor1, dim=-1)
      norm2 = torch.linalg.norm(tensor2, dim=-1)

      eps = 1e-8  # Small epsilon value
      norm1 = torch.clamp(norm1, min=eps)
      norm2 = torch.clamp(norm2, min=eps)

      cosine_sim = dot_product / (norm1 * norm2)
      return cosine_sim

    def kl_divergence(self, p, q, tt):
      try:
        return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
      except:
        return sum(p[i] * log2(p[i]/(q[i] + 1e-12)) for i in range(len(p)))

    def multilabel_cl(self, feats, labels, random_anchor):
        labels = labels.cpu()
        index_sum = labels.sum(dim=1, keepdim=True)
        matmul = torch.matmul(labels, labels[random_anchor])
        selected = np.where(matmul == index_sum[random_anchor])[0]
        mask = selected != random_anchor
        positives = selected[mask]
        negatives = np.where(matmul == 0)[0]
        anchor = feats[random_anchor]

        anchor_dot_positives = 0
        positives_tensor = []
        for positive in positives:
          anchor_dot_positives = anchor_dot_positives + self.cosine_similarity(anchor, feats[positive])
          temp = self.cosine_similarity(anchor, feats[positive])
          positives_tensor.append(max(1e-12, min(temp, 1)))
        positive_cross_entropy = self.cross_entropy(torch.tensor(positives_tensor, requires_grad=True), torch.ones(len(positives_tensor)))
        negative_matmul = torch.matmul(anchor, feats[negatives].permute(1,0))
        negative_cross_entropy = self.alpha * self.kl_divergence(matmul, torch.zeros(len(matmul)), "neg")
        return positive_cross_entropy

    def scmc(self, feats, labels, random_anchor):
        temp_feature =  torch.matmul(feats[random_anchor], feats.permute(1,0))
        zeros = torch.zeros(feats.size(0))
        logit = torch.divide(temp_feature, self.temp)

        labels = labels.cpu()

        index_sum = labels.sum(dim=1, keepdim=True)
        matmul = torch.matmul(labels, labels[random_anchor])

        selected = np.where(matmul == index_sum[random_anchor])[0]
        zeros[selected] = 1
        return self.cross_entropy(temp_feature.to(device), zeros.to(device))

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)

        random_anchor = random.randint(0, normed_cls_feats.size(0) - 1)
        cl_loss = self.alpha * self.multilabel_cl(normed_cls_feats, targets, random_anchor)
        return cl_loss