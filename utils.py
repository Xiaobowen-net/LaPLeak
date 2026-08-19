import argparse
import csv
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_1
import copy
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from piq import ssim,psnr,LPIPS
import math


# ============================================================================
# EMA 类原型（contrastive/prototype 思想的有效落点）
# 维护 target 特征空间(teacher 的输入)上的每类 EMA 原型，用高置信样本更新；
# 再用最近原型(余弦)给出一个 prototype-based 软标签，blend 进蒸馏目标，
# 纠正 teacher 的噪声伪标签（尤其少数类），直接提升 label-inference 指标，
# 且不触碰 pseudo_bottom 的对抗对齐（避免 b3t3 上观察到的失稳）。
# ============================================================================
_PROTO_STATE = {}


def update_and_score_prototypes(feat, pseudo_labels, conf,
                                num_classes=10, ema=0.9, temp=0.1,
                                conf_th=0.7, key='default'):
    """feat:[B,D] (4D 会先 GAP)；返回 prototype 软标签 [B,C] 与已初始化掩码。"""
    if feat.dim() == 4:
        feat = feat.mean(dim=(2, 3))
    device = feat.device
    D = feat.size(1)

    state = _PROTO_STATE.get(key)
    if state is None or state['proto'].size(1) != D or state['proto'].device != device:
        state = {
            'proto': torch.zeros(num_classes, D, device=device),
            'init': torch.zeros(num_classes, dtype=torch.bool, device=device),
        }
        _PROTO_STATE[key] = state
    proto = state['proto']
    inited = state['init']

    with torch.no_grad():
        for c in range(num_classes):
            mask = (pseudo_labels == c) & (conf > conf_th)
            if mask.sum() >= 1:
                mean_c = feat[mask].mean(dim=0)
                if inited[c]:
                    proto[c] = ema * proto[c] + (1.0 - ema) * mean_c
                else:
                    proto[c] = mean_c
                    inited[c] = True

    feat_n = F.normalize(feat, dim=1)
    proto_n = F.normalize(proto, dim=1)
    sims = feat_n @ proto_n.t()                              # [B, C] cosine
    sims = sims.masked_fill(~inited.unsqueeze(0), -1e4)      # 未初始化类屏蔽
    proto_probs = torch.softmax(sims / temp, dim=1)
    return proto_probs, inited


def reset_prototype_state():
    _PROTO_STATE.clear()


class Crop(torch.nn.Module):

    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):

        return F_1.crop(img, self.top, self.left, self.height, self.width)


class DeNormalize(nn.Module):               
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Normalized Tensor image.

        Returns:
            Tensor: Denormalized Tensor.
        """
        return self._denormalize(tensor)

    def _denormalize(self, tensor):
        tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        # tensor.sub_(mean).div_(std)
        tensor.mul_(std).add_(mean)

        return tensor
def gradient_penalty(discriminator, x, x_gen,device):
    """
    Args:
        x
        x_gen

    Returns:
        d_regularizer
    """
    epsilon = torch.rand([x.shape[0], 1, 1, 1]).to(device)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    from torch.autograd import grad
    d_hat = discriminator(x_hat)
    gradients = grad(outputs=d_hat, inputs=x_hat,
                    grad_outputs=torch.ones_like(d_hat).to(device),
                    retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0),  -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1)**2).mean()

    return penalty

def Pre_training(pseudo_top_teacher,teacher_optimizer, target_splitnn, target_invmodel, target_invmodel_optimizer,
                   pseudo_bottom, pseudo_top, pseudo_invmodel, pseudo_invmodel_optimizer,pseudo_bottom_optimizer, pseudo_top_optimizer,
                    discriminator, discriminator_optimizer,
                   target_data, target_label, shadow_data, shadow_label, device, n, iteration, a):
    

    target_splitnn.train()
    target_invmodel.train()

################################################### 拆分网络训练 #############################
    target_data = target_data.to(device)
    target_label = target_label.to(device)
    target_splitnn.zero_grads()
    target_splitnn_output = target_splitnn(target_data)

    target_splitnn_intermidiate = target_splitnn.intermidiate_to_server.detach()
    target_intermidiate_to_top = target_splitnn.intermidiate_to_top.detach()

    target_splitnn_celoss = F.cross_entropy(target_splitnn_output,target_label)

    target_splitnn_celoss.backward()
    target_splitnn.backward()
    target_splitnn.step()

    for para in target_splitnn.client.client_model.parameters():
        para.requires_grad = False
    for para in target_splitnn.server.server_model.parameters():
        para.requires_grad = False
    for para in target_splitnn.top.top_model.parameters():
        para.requires_grad = False

    target_invmodel_optimizer.zero_grad()
    target_inv_input = target_splitnn_intermidiate.detach()
    target_inv_output = target_invmodel(target_inv_input) 
    target_inv_loss = F.mse_loss(target_inv_output, target_data)
    target_inv_loss.backward()
    target_invmodel_optimizer.step()


    pseudo_bottom.train()
    pseudo_top.train()
    pseudo_top_teacher.train()
    pseudo_invmodel.train()
    discriminator.train()

    shadow_data = shadow_data.to(device)
    shadow_label = shadow_label.to(device)

    pseudo_bottom_optimizer.zero_grad()

    pseudo_output = pseudo_bottom(shadow_data)


    d_input_pseudo = pseudo_output
    d_output_pseudo = discriminator(d_input_pseudo)

    pseudo_d_loss = torch.mean(d_output_pseudo)
    pseudo_d_loss.backward()
    pseudo_bottom_optimizer.step()
    for para in pseudo_bottom.parameters():
        para.requires_grad = False

    pseudo_invmodel_input = pseudo_output.detach()
    pseudo_invmodel_output = pseudo_invmodel(pseudo_invmodel_input)
    pseudo_inv_loss = F.mse_loss(pseudo_invmodel_output,shadow_data)
    pseudo_invmodel_optimizer.zero_grad()
    pseudo_inv_loss.backward()
    pseudo_invmodel_optimizer.step()

    discriminator_optimizer.zero_grad()
    pseudo_discriminator_output = pseudo_output.detach()
    target_client_output = target_splitnn_intermidiate.detach()
    adv_target_logits = discriminator(target_client_output)
    adv_ae_logits = discriminator(pseudo_discriminator_output)
    loss_discr_true = torch.mean(adv_target_logits)
    loss_discr_fake = -torch.mean(adv_ae_logits)
    vanila_D_loss = loss_discr_true + loss_discr_fake
    D_loss = vanila_D_loss + 10 * gradient_penalty( discriminator,pseudo_output.detach(), target_splitnn_intermidiate.detach(), device)
    D_loss.backward()
    discriminator_optimizer.step()


    ########################## 训练 Top ##########################
    pseudo_top_optimizer.zero_grad()
    teacher_optimizer.zero_grad()
    server_to_pseudotop = target_splitnn.server(pseudo_output.detach())

    pseudo_label = pseudo_top(server_to_pseudotop.detach())
    pseudo_celoss = F.cross_entropy(pseudo_label, shadow_label)
    pseudo_celoss.backward()
    pseudo_top_optimizer.step()

    teacher_label = pseudo_top_teacher(server_to_pseudotop.detach())
    teacher_celoss = F.cross_entropy(teacher_label, shadow_label)
    teacher_celoss.backward()
    teacher_optimizer.step()

    for para in target_splitnn.client.client_model.parameters():
        para.requires_grad = True
    for para in target_splitnn.server.server_model.parameters():
        para.requires_grad = True
    for para in target_splitnn.top.top_model.parameters():
        para.requires_grad = True
    for para in pseudo_bottom.parameters():
        para.requires_grad = True
    for para in pseudo_top.parameters():
        para.requires_grad = True
    for para in pseudo_top_teacher.parameters():
        para.requires_grad = True
    return target_splitnn_intermidiate.detach(), target_intermidiate_to_top


def pseudo_training(pseudo_top_teacher,teacher_optimizer,target_splitnn, target_invmodel, target_invmodel_optimizer,
                    pseudo_bottom, pseudo_top, pseudo_invmodel, pseudo_invmodel_optimizer, pseudo_bottom_optimizer,
                    pseudo_top_optimizer,
                    discriminator, discriminator_optimizer,
                    target_data, target_label, shadow_data, shadow_label, device, n, iteration, a,private_distributed, aux_dist,
                    mmd_weight=0.5,
                    ldfa_w=1.0, ldfa_temp=0.5, ldfa_conf=0.4,
                    shift_adaptive=True, shift_gate_s0=0.15, cal_coef=0.5,
                    proto_refine_w=0.3, proto_ema=0.9, proto_temp=0.1,
                    proto_conf=0.7, proto_warmup=2000):
    target_splitnn.train()
    target_invmodel.train()

    ################################################### 拆分网络训练 #############################
    target_data = target_data.to(device)
    target_label = target_label.to(device)
    target_splitnn.zero_grads()
    target_splitnn_output = target_splitnn(target_data)

    target_splitnn_intermidiate = target_splitnn.intermidiate_to_server.detach()
    target_intermidiate_to_top = target_splitnn.intermidiate_to_top.detach()

    target_splitnn_celoss = F.cross_entropy(target_splitnn_output, target_label)

    target_splitnn_celoss.backward()
    target_splitnn.backward()
    target_splitnn.step()

    for para in target_splitnn.client.client_model.parameters():
        para.requires_grad = False
    for para in target_splitnn.server.server_model.parameters():
        para.requires_grad = False
    for para in target_splitnn.top.top_model.parameters():
        para.requires_grad = False

    pseudo_bottom.train()
    discriminator.train()


    shadow_data = shadow_data.to(device)
    shadow_label = shadow_label.to(device)

    pseudo_bottom_optimizer.zero_grad()

#########

    aux_dist_t = torch.as_tensor(aux_dist, device=device, dtype=torch.float32)

    # === label-shift 自适应门控 ===========================================
    # 估计当前 shift 强度 s = TV(pi, aux) ∈ [0,1]。无偏移（均匀）时 s≈0（仅 EM 噪声），
    # 门控 gate→0，自动关闭 logit 校准与先验 marginal 约束 —— 这正是均匀分布下变差的根因：
    # 校准/先验把 EM 噪声当成真实偏移注入，破坏了本就匹配的 teacher。
    # 重度偏移时 s 大，gate→1，完整保留原有的 label-shift 校正思想（行为与基线一致）。
    if shift_adaptive:
        shift_strength = 0.5 * (private_distributed.detach() - aux_dist_t).abs().sum()
        shift_gate = (shift_strength / (shift_gate_s0 + 1e-8)).clamp(0.0, 1.0)
    else:
        shift_gate = torch.tensor(1.0, device=device)

    with torch.no_grad():
        # 关键：teacher 在 target 特征上做前向时必须用 eval 模式，
        # 否则 BN running stats 会被 target 分布污染（teacher 是在 shadow 上训的）
        pseudo_top_teacher.eval()
        teacher_logits = pseudo_top_teacher(target_intermidiate_to_top.detach())  # [B, C]
        pseudo_top_teacher.train()

        log_prior_shift = torch.log(private_distributed + 1e-8) - \
                          torch.log(aux_dist_t + 1e-8)

        # 校准系数随 shift 门控缩放：无偏移→≈0（恒等校准），重度偏移→cal_coef
        calibrated_teacher_logits = teacher_logits + (cal_coef * shift_gate) * log_prior_shift.unsqueeze(0)


        calibrated_probs = torch.softmax(calibrated_teacher_logits, dim=1)


    calibrated_conf, calibrated_labels = torch.max(calibrated_probs, dim=1)
    # change_rate = (calibrated_labels != teacher_label).float().mean()
    # print(change_rate)
    #

    # === EMA 类原型精修蒸馏目标（contrastive 思想的有效落点）====================
    # 用高置信样本在 target 特征空间维护每类 EMA 原型，再用最近原型(余弦)软标签
    # 与 teacher 校准概率融合，纠正噪声伪标签。warmup 后才启用，让原型先稳定。
    refined_probs = calibrated_probs
    proto_active = 0.0
    if proto_refine_w > 0:
        with torch.no_grad():
            proto_probs, proto_inited = update_and_score_prototypes(
                target_intermidiate_to_top.detach(), calibrated_labels, calibrated_conf,
                num_classes=10, ema=proto_ema, temp=proto_temp, conf_th=proto_conf, key='target_top')
            if n > proto_warmup:
                # 逐样本门控：只有当某样本的预测类原型已就绪时才精修该样本，
                # 否则保留 teacher 校准概率。这样重度偏移下少数类原型缺失也不会误导。
                ready = proto_inited[calibrated_labels].float().unsqueeze(1)   # [B,1]
                blend = proto_refine_w * ready
                refined_probs = (1.0 - blend) * calibrated_probs + blend * proto_probs

                
                proto_active = float(ready.mean())

    pseudo_output = pseudo_bottom(shadow_data)
    d_output_pseudo = discriminator(pseudo_output)
    # ldfa = compute_ldfa(pseudo_output.mean(dim=(2, 3)),target_splitnn_intermidiate.detach().mean(dim=(2, 3)),shadow_label,calibrated_labels,calibrated_conf,None, eps=1e-8)

    # pseudo_d_loss = torch.mean(d_output_pseudo) # + 0.3 * ldfa
    # ldfa = proto(pseudo_output, target_splitnn_intermidiate, shadow_label, calibrated_labels, calibrated_conf)

    # 类条件对比对齐（保留原思想，但留在 pseudo_bottom 上时是惰性的 ~ln9 chance）：
    # 经验证调低温度只会把噪声梯度灌进 pseudo_bottom 进而拖垮 teacher（shift 85->77），故 ldfa_temp 保持 0.5
    # 的“无害惰性”值。对比/原型思想真正起作用的位置是下游对蒸馏伪标签的 EMA 原型精修（见上文 refined_probs）。
    ldfa = compute_ldfa_infonce(pseudo_output,  target_splitnn_intermidiate, shadow_label, calibrated_labels, calibrated_conf, num_classes=10, temperature=ldfa_temp, conf_threshold=ldfa_conf,min_per_class=2,eps=1e-8)

    # 新增：MK-MMD 全局特征对齐（GAP 后比较通道分布），与对抗 + InfoNCE 互补
    if mmd_weight > 0:
        mmd_loss = compute_mmd_alignment(pseudo_output, target_splitnn_intermidiate.detach())
    else:
        mmd_loss = torch.tensor(0.0, device=device, dtype=pseudo_output.dtype)

    pseudo_d_loss = torch.mean(d_output_pseudo) + ldfa_w * ldfa # + mmd_weight * mmd_loss

    if n % 2000 == 0:
        print(f"  [shift-diag] gate={float(shift_gate):.3f} adv={float(torch.mean(d_output_pseudo)):.3f} ldfa={float(ldfa):.3f} proto_active={proto_active:.0f}")

    pseudo_d_loss.backward()
    pseudo_bottom_optimizer.step()

    for para in pseudo_bottom.parameters():
        para.requires_grad = False

    pseudo_invmodel_input = pseudo_output.detach()
    pseudo_invmodel_output = pseudo_invmodel(pseudo_invmodel_input)
    pseudo_inv_loss = F.mse_loss(pseudo_invmodel_output, shadow_data)
    pseudo_invmodel_optimizer.zero_grad()
    pseudo_inv_loss.backward()
    pseudo_invmodel_optimizer.step()

    discriminator_optimizer.zero_grad()
    pseudo_discriminator_output = pseudo_output.detach()
    target_client_output = target_splitnn_intermidiate.detach()
    adv_target_logits = discriminator(target_client_output)
    adv_ae_logits = discriminator(pseudo_discriminator_output)
    loss_discr_true = torch.mean(adv_target_logits)
    loss_discr_fake = -torch.mean(adv_ae_logits)
    vanila_D_loss = loss_discr_true + loss_discr_fake
    D_loss = vanila_D_loss + 10 * gradient_penalty(discriminator, pseudo_output.detach(),
                                                     target_splitnn_intermidiate.detach(), device)
    D_loss.backward()
    discriminator_optimizer.step()

    ########################## 训练 Top ##########################
    pseudo_top.train()
    for para in pseudo_top.parameters():
        para.requires_grad = True

    pseudo_top_optimizer.zero_grad()

    server_to_pseudotop = target_splitnn.server(pseudo_output.detach())
    pseudo_zhen_label = pseudo_top(server_to_pseudotop)


    ######### 硬标签 ####
    student_logits = pseudo_top(target_intermidiate_to_top.detach())  # 重新 forward
    # pseudo_loss_vec = F.cross_entropy(student_logits, calibrated_labels.detach(), reduction='none')
    # weights = calibrated_conf.detach()
    # weighted_loss = (pseudo_loss_vec * weights).sum() / (weights.sum() + 1e-8)
###############

######### KL ######
    student_log_probs = F.log_softmax(student_logits, dim=1)
    loss_target_vec = F.kl_div(
        student_log_probs,
        refined_probs.detach(),  # teacher target (经 EMA 原型精修), stopgrad
        reduction='none'
    ).sum(dim=1)
    weights = calibrated_conf.detach()
    # class-balanced reweighting：给少数类的高 conf 样本额外加权
    # inv_freq = 1.0 / (private_distributed[calibrated_labels] + 1e-3)
    # inv_freq = inv_freq / inv_freq.mean()
    # weights = calibrated_conf.detach() * inv_freq

    loss_target = (loss_target_vec * weights).sum() / (weights.sum() + 1e-8)

################################
    # 可选：加一个全局边缘分布约束，避免 student 又回到 aux prior
    student_marginal = torch.softmax(student_logits, dim=1).mean(dim=0)
    # 先验 marginal 约束同样按 shift 门控：无偏移时关闭，避免把 EM 噪声当真实先验强加给 student。
    loss_prior = shift_gate * F.kl_div(
        (student_marginal + 1e-8).log(),
        private_distributed.detach(),
        reduction='sum'
    )

    # === 新增：在 shadow CE 上做 label-shift 重要性加权 ===
    # 目的：shadow 是 aux_dist (均匀)，但我们希望 student 行为接近在 private_dist 上训出来的。
    # importance weight  w_i = private_dist[y_i] / aux_dist[y_i]

    # aux_dist_t = torch.as_tensor(aux_dist, device=device, dtype=torch.float32)
    # iw = (private_distributed.detach() + 1e-8) / (aux_dist_t + 1e-8)
    # iw = iw / iw.mean().clamp(min=1e-8)            # 归一化保持 loss 尺度
    # iw = iw.clamp(0.1, 10.0)                       # 截断稳定性
    # sample_iw = iw[shadow_label]
    # ce_vec = F.cross_entropy(pseudo_zhen_label, shadow_label, reduction='none')
    # weighted_ce = (ce_vec * sample_iw).mean()

    ce_vec = F.cross_entropy(pseudo_zhen_label, shadow_label, reduction='none')
    weighted_ce = ce_vec.mean()

    # student_celoss = F.cross_entropy(pseudo_zhen_label, shadow_label) + weighted_loss + loss_prior
    student_celoss = weighted_ce + loss_target + loss_prior


    student_celoss.backward()
    pseudo_top_optimizer.step()
    for para in pseudo_top.parameters():
        para.requires_grad = False

    ###############
    pseudo_top_teacher.train()
    teacher_optimizer.zero_grad()
    teacher_shadow_logits = pseudo_top_teacher(server_to_pseudotop.detach())
    #######
    # teacher_target_logits = pseudo_top_teacher(target_intermidiate_to_top.detach())
    # teacher_target_probs = torch.softmax(teacher_target_logits, dim=1)
    #
    # entropy = -(teacher_target_probs * (teacher_target_probs + 1e-8).log()).sum(dim=1)
    # sample_w = 1.0 - entropy / math.log(teacher_target_probs.size(1))
    # sample_w = sample_w / (sample_w.sum() + 1e-8)
    #
    # p_bar = (teacher_target_probs * sample_w.unsqueeze(1)).sum(dim=0)
    # p_bar = p_bar / (p_bar.sum() + 1e-8)
    # beta = 0.05
    # loss_teacher_marg = F.kl_div(
    #     (p_bar + 1e-8).log(),
    #     private_distributed.detach(),
    #     reduction='sum'
    # )
    # #
    # with torch.no_grad():
    #     # 获取此时 Student 对目标域的最新预测（已包含分布倾向）
    #     student_target_logits = pseudo_top(target_intermidiate_to_top.detach())
    #     student_target_probs = torch.softmax(student_target_logits, dim=1)
    #
    # loss_reverse_kd = F.kl_div(
    #     F.log_softmax(teacher_target_logits, dim=1),
    #     student_target_probs,
    #     reduction='batchmean'
    # )

    teacher_loss = F.cross_entropy(teacher_shadow_logits, shadow_label)

    teacher_loss.backward()
    teacher_optimizer.step()

    for para in target_splitnn.client.client_model.parameters():
        para.requires_grad = True
    for para in target_splitnn.server.server_model.parameters():
        para.requires_grad = True
    for para in target_splitnn.top.top_model.parameters():
        para.requires_grad = True
    for para in pseudo_bottom.parameters():
        para.requires_grad = True
    for para in pseudo_top.parameters():
        para.requires_grad = True

    return target_splitnn_intermidiate.detach(), target_intermidiate_to_top

"Train the Target Model and the Shadow Model"
def cla_train(splitnn, invmodel, invmodel_optimizer, target_data, target_label, dataloader, epoch, print_freq, device):

    splitnn.train()
    invmodel.train()

    data = target_data.to(device)
    target = target_label.to(device)

    splitnn.zero_grads()
    output = splitnn(data)

    celoss = F.cross_entropy(output,target)
    celoss.backward()

    splitnn.backward()
    gridients = splitnn.server.grad_to_client
    splitnn.step()

    if epoch % print_freq == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}'.format(
            epoch, epoch * len(data), len(dataloader.dataset),
            100. * epoch / len(dataloader), celoss.item(),))
    return epoch, gridients
    
def proto(source_feat,          # [Bs, D]
    target_feat,          # [Bt, D]
    source_label,         # [Bs]
    calibrated_labels,    # [Bt]
    target_weight):
    # 用高置信样本估 per-class prototype
    proto_t = {}
    for c in range(10):
        mask_c = (calibrated_labels == c) & (target_weight > 0.4)  # 高 conf 才参与
        if mask_c.sum() >= 2:
            proto_t[c] = target_feat[mask_c].mean(dim=0)

    # shadow 端按真实标签算到对应 prototype 的距离
    loss = 0
    count = 0
    for c in proto_t:
        mask_s = (source_label == c)
        if mask_s.sum() > 0:
            loss = loss + F.mse_loss(source_feat[mask_s], proto_t[c].expand_as(source_feat[mask_s]))
            count += 1
    proto = loss / max(count, 1)
    return proto





def compute_ldfa(
    source_feat,          # [Bs, D]
    target_feat,          # [Bt, D]
    source_label,         # [Bs]
    calibrated_labels,    # [Bt]
    target_weight,        # [Bt]
    source_weight=None,   # [Bs] or None
    eps=1e-8
):
    device = source_feat.device
    Bs = source_feat.size(0)

    if source_weight is None:
        source_weight = torch.ones(Bs, device=device, dtype=source_feat.dtype)

    # 1. pairwise distance Phi(x_i^s, x_j^t)
    dist_mat = torch.cdist(source_feat, target_feat, p=2)   # [Bs, Bt]

    # 2. pairwise weighted term: sqrt(w_i^s * w_j^t * Phi)
    pair_term = torch.sqrt(
        source_weight.unsqueeze(1) * target_weight.unsqueeze(0) * dist_mat + eps
    )   # [Bs, Bt]

    # 3. same / diff masks
    same_mask = source_label.unsqueeze(1).eq(calibrated_labels.unsqueeze(0))   # [Bs, Bt]
    diff_mask = ~same_mask

    N_same = same_mask.sum()
    N_diff = diff_mask.sum()

    # 避免某个 batch 恰好没有 same 或 diff 配对
    if N_same.item() == 0 or N_diff.item() == 0:
        return torch.tensor(0.0, device=device, dtype=source_feat.dtype)

    same_avg = pair_term[same_mask].mean()
    diff_avg = pair_term[diff_mask].mean()

    ldfa = same_avg / (diff_avg + eps)
    return ldfa

"Train the Accuracy of Target Model and Shadow Model"
def cla_target_test(target_splitnn, test_loader, device):
    target_splitnn.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = target_splitnn(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Target Average Loss: {:.4f}, Target Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

def cla_pseudo_test(target_splitnn, pseudo_top, test_loader, device):
    target_splitnn.eval()
    pseudo_top.eval()

    pseudo_test_loss = 0
    target_test_loss = 0
    target_correct = 0
    pseudo_correct = 0
    fidelity = 0
    both_correct_and_same= 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            target_output = target_splitnn(data)
            top_data = target_splitnn.server(target_splitnn.client(data))

            pseudo_output = pseudo_top(top_data)

            target_test_loss += F.cross_entropy(target_output, target, reduction='sum').item()
            target_pred = target_output.argmax(dim=1, keepdim=True)
            target_correct += target_pred.eq(target.view_as(target_pred)).sum().item()


            pseudo_test_loss += F.cross_entropy(pseudo_output, target, reduction='sum').item()
            pseudo_pred = pseudo_output.argmax(dim=1, keepdim=True)
            pseudo_correct += pseudo_pred.eq(target.view_as(pseudo_pred)).sum().item()



            both_correct_and_same += (target_pred.eq(pseudo_pred.view_as(target_pred)) &target_pred.eq(target.view_as(target_pred))).sum().item()

            fidelity += target_pred.eq(pseudo_pred.view_as(target_pred)).sum().item()

    target_test_loss /= len(test_loader.dataset)

    pseudo_test_loss /= len(test_loader.dataset)

    # correct = round(correct / len(test_loader.dataset),2)

    print('\nTarget Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        target_test_loss, target_correct, len(test_loader.dataset),
        100. * target_correct / len(test_loader.dataset)))

    print('\nPseudo Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        pseudo_test_loss, pseudo_correct, len(test_loader.dataset),
        100. * pseudo_correct / len(test_loader.dataset)))

    print('\nFidelity: {:.0f}%, Both_correct_and_same: {:.0f}%\n'.format(
        100. * fidelity / len(test_loader.dataset), 100. * both_correct_and_same / len(test_loader.dataset)))

    return {
        'target_loss': target_test_loss,
        'target_correct': 100. * target_correct / len(test_loader.dataset),
        'pseudo_loss': pseudo_test_loss,
        'pseudo_correct': 100. * pseudo_correct / len(test_loader.dataset),
        'fidelity': 100. * fidelity / len(test_loader.dataset)
    }

# def discriminator_train()

def attack_test(target_splitnn, target_invmodel, pseudo_invmodel, device, target_bottom, pseudo_bottom, train_dataloader,n, iteration, num_classes=10):
    target_splitnn.eval()
    target_invmodel.eval()
    pseudo_invmodel.eval()

    # 总体累加器
    target_loss = 0
    pseudo_loss = 0
    target_ssim = 0
    target_psnr = 0
    pseudo_ssim = 0
    pseudo_psnr = 0
    target_pseudo_mse = 0

    # 按类别累加器
    cls_count = [0] * num_classes
    cls_target_loss = [0.0] * num_classes      # per-pixel MSE 和
    cls_pseudo_loss = [0.0] * num_classes
    cls_target_ssim = [0.0] * num_classes
    cls_pseudo_ssim = [0.0] * num_classes
    cls_target_psnr = [0.0] * num_classes
    cls_pseudo_psnr = [0.0] * num_classes
    cls_intermid_mse = [0.0] * num_classes     # per-feature MSE 和

    target_model_ = copy.deepcopy(target_bottom)
    pseudo_model_ = copy.deepcopy(pseudo_bottom)
    target_model_.eval()
    pseudo_model_.eval()

    denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    with torch.no_grad():
        for target_data, target_label in train_dataloader:
            dataset_shape = target_data[0].shape

            target_data = target_data.to(device)
            target_label = target_label.to(device)

            target_output = target_splitnn(target_data)
            target_splitnn_intermidiate = target_splitnn.client(target_data)
            target_output = target_splitnn_intermidiate.detach()

            pseudo_inv_output = pseudo_invmodel(target_output)
            target_inv_output = target_invmodel(target_output)

            # 总体（保持与原逻辑一致）
            pseudo_loss += F.mse_loss(target_data, pseudo_inv_output, reduction='sum').item()
            target_loss += F.mse_loss(target_data, target_inv_output, reduction='sum').item()

            pseudo_inv_output_ = denorm(pseudo_inv_output.detach().clone())
            target_inv_output_ = denorm(target_inv_output.detach().clone())
            original_data = denorm(target_data.clone())

            ts_ssim = ssim(original_data, target_inv_output_, reduction='none')   # [B]
            ps_ssim = ssim(original_data, pseudo_inv_output_, reduction='none')   # [B]
            ts_psnr = psnr(original_data, target_inv_output_, reduction='none')   # [B]
            ps_psnr = psnr(original_data, pseudo_inv_output_, reduction='none')   # [B]

            target_ssim += ts_ssim.sum().item()
            target_psnr += ts_psnr.sum().item()
            pseudo_ssim += ps_ssim.sum().item()
            pseudo_psnr += ps_psnr.sum().item()

            pseudo_inter = pseudo_model_(target_data)
            target_inter = target_model_(target_data)
            target_pseudo_mse += F.mse_loss(pseudo_inter, target_inter, reduction='sum').item()

            # 按类别（per-sample MSE，已在 CHW 上取平均）
            ts_mse_per = F.mse_loss(target_data, target_inv_output, reduction='none').mean(dim=(1, 2, 3))  # [B]
            ps_mse_per = F.mse_loss(target_data, pseudo_inv_output, reduction='none').mean(dim=(1, 2, 3))  # [B]
            inter_mse_per = F.mse_loss(pseudo_inter, target_inter, reduction='none').mean(dim=(1, 2, 3))   # [B]

            for c in range(num_classes):
                mask = (target_label == c)
                n_c = mask.sum().item()
                if n_c == 0:
                    continue
                cls_count[c] += n_c
                cls_target_loss[c] += ts_mse_per[mask].sum().item()
                cls_pseudo_loss[c] += ps_mse_per[mask].sum().item()
                cls_target_ssim[c] += ts_ssim[mask].sum().item()
                cls_pseudo_ssim[c] += ps_ssim[mask].sum().item()
                cls_target_psnr[c] += ts_psnr[mask].sum().item()
                cls_pseudo_psnr[c] += ps_psnr[mask].sum().item()
                cls_intermid_mse[c] += inter_mse_per[mask].sum().item()

    N = len(train_dataloader.dataset)
    pixel_factor = dataset_shape[0] * dataset_shape[1] * dataset_shape[2]
    inter_factor = pseudo_inter.shape[1] * pseudo_inter.shape[2] * pseudo_inter.shape[3]

    pseudo_loss = pseudo_loss / (N * pixel_factor)
    target_loss = target_loss / (N * pixel_factor)
    target_pseudo_mse = target_pseudo_mse / (N * inter_factor)
    target_ssim /= N
    target_psnr /= N
    pseudo_ssim /= N
    pseudo_psnr /= N

    print('Train Iteration: [{}/{} ({:.0f}%)]'.format(n, iteration, 100. * n / iteration))
    print('target_ssim: {:.6f}  pseudo_ssim: {:.6f}  target_psnr: {:.6f}  pseudo_psnr: {:.6f}  target_mse: {:.6f}  pseudo_mse: {:.6f}  intermidiate_mse:  {:.6f}'.format(
        target_ssim, pseudo_ssim, target_psnr, pseudo_psnr, target_loss, pseudo_loss, target_pseudo_mse))

    # ===== 按类别打印 =====
    print('\n========== Per-class reconstruction metrics ==========')
    header = '{:<3s} {:>5s} | {:>8s} {:>8s} | {:>8s} {:>8s} | {:>9s} {:>9s} | {:>9s}'.format(
        'c', 'n', 'tgt_ssim', 'psd_ssim', 'tgt_psnr', 'psd_psnr', 'tgt_mse', 'psd_mse', 'inter_mse')
    print(header)
    print('-' * len(header))
    per_class = {}
    for c in range(num_classes):
        cnt = cls_count[c]
        if cnt == 0:
            print('{:<3d} {:>5d} | (no samples)'.format(c, cnt))
            per_class[c] = None
            continue
        tgt_ssim_c = cls_target_ssim[c] / cnt
        psd_ssim_c = cls_pseudo_ssim[c] / cnt
        tgt_psnr_c = cls_target_psnr[c] / cnt
        psd_psnr_c = cls_pseudo_psnr[c] / cnt
        tgt_mse_c = cls_target_loss[c] / cnt
        psd_mse_c = cls_pseudo_loss[c] / cnt
        inter_mse_c = cls_intermid_mse[c] / cnt
        print('{:<3d} {:>5d} | {:>8.4f} {:>8.4f} | {:>8.4f} {:>8.4f} | {:>9.5f} {:>9.5f} | {:>9.5f}'.format(
            c, cnt, tgt_ssim_c, psd_ssim_c, tgt_psnr_c, psd_psnr_c, tgt_mse_c, psd_mse_c, inter_mse_c))
        per_class[c] = {
            'count': cnt,
            'target_ssim': tgt_ssim_c, 'pseudo_ssim': psd_ssim_c,
            'target_psnr': tgt_psnr_c, 'pseudo_psnr': psd_psnr_c,
            'target_mse': tgt_mse_c, 'pseudo_mse': psd_mse_c,
            'intermidiate_mse': inter_mse_c,
        }
    print('=' * len(header))

    return {
        'target_ssim': target_ssim,
        'pseudo_ssim': pseudo_ssim,
        'target_psnr': target_psnr,
        'pseudo_psnr': pseudo_psnr,
        'target_loss': target_loss,
        'pseudo_loss': pseudo_loss,
        'target_pseudo_mse': target_pseudo_mse,
        'per_class': per_class
    }



def attack_cla_train(target_splitnn, pseudo_top, train_dataloader, device, num_classes):
    target_splitnn.eval()
    pseudo_top.eval()
    # 按类别统计
    class_total = [0 for _ in range(num_classes)]
    class_target_correct = [0 for _ in range(num_classes)]
    class_pseudo_correct = [0 for _ in range(num_classes)]
    class_fidelity = [0 for _ in range(num_classes)]
    class_both_correct_and_same = [0 for _ in range(num_classes)]
    all_preds = []
    all_labels = []
    pseudo_test_loss = 0
    target_test_loss = 0
    target_correct = 0
    pseudo_correct = 0
    fidelity = 0
    both_correct_and_same = 0
    with torch.no_grad():
        for data, target in train_dataloader:
            data = data.to(device)
            target = target.to(device)

            target_output = target_splitnn(data)

            top_data = target_splitnn.server(target_splitnn.client(data))

            pseudo_output = pseudo_top(top_data)

            target_test_loss += F.cross_entropy(target_output, target, reduction='sum').item()
            target_pred = target_output.argmax(dim=1, keepdim=True)
            target_correct += target_pred.eq(target.view_as(target_pred)).sum().item()

            pseudo_test_loss += F.cross_entropy(pseudo_output, target, reduction='sum').item()
            pseudo_pred = pseudo_output.argmax(dim=1, keepdim=True)
            pseudo_correct += pseudo_pred.eq(target.view_as(pseudo_pred)).sum().item()

            all_preds.append(pseudo_pred.cpu())
            all_labels.append(target.cpu())

            both_correct_and_same += (target_pred.eq(pseudo_pred.view_as(target_pred)) & target_pred.eq(
                target.view_as(target_pred))).sum().item()

            fidelity += target_pred.eq(pseudo_pred.view_as(target_pred)).sum().item()

            # 按类别统计
            for c in range(num_classes):
                mask = (target == c)  # 真实标签为 c 的样本

                class_total[c] += mask.sum().item()

                target_pred_flat = target_pred.squeeze(1)
                pseudo_pred_flat = pseudo_pred.squeeze(1)

                class_target_correct[c] += (target_pred_flat[mask] == target[mask]).sum().item()
                class_pseudo_correct[c] += (pseudo_pred_flat[mask] == target[mask]).sum().item()
                class_fidelity[c] += (target_pred_flat[mask] == pseudo_pred_flat[mask]).sum().item()
                class_both_correct_and_same[c] += (
                        (target_pred_flat[mask] == pseudo_pred_flat[mask]) &
                        (target_pred_flat[mask] == target[mask])
                ).sum().item()


    target_test_loss /= len(train_dataloader.dataset)

    pseudo_test_loss /= len(train_dataloader.dataset)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        zero_division=0
    )

    # correct = round(correct / len(test_loader.dataset),2)
    ##########################################             每类指标            ############################################
    per_class_target_acc = []
    per_class_pseudo_acc = []
    per_class_fidelity = []
    per_class_both_correct_and_same = []
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_target_acc.append(class_target_correct[c] / class_total[c])
            per_class_pseudo_acc.append(class_pseudo_correct[c] / class_total[c])
            per_class_fidelity.append(class_fidelity[c] / class_total[c])
            per_class_both_correct_and_same.append(class_both_correct_and_same[c] / class_total[c])
        else:
            per_class_target_acc.append(0.0)
            per_class_pseudo_acc.append(0.0)
            per_class_fidelity.append(0.0)
            per_class_both_correct_and_same.append(0.0)
    for c in range(10):
        print(f"class {c}:")
        print("total =", class_total[c])
        print('target_acc {:.0f}%'.format(100. * per_class_target_acc[c]))
        print('pseudo_acc {:.0f}%'.format(100. * per_class_pseudo_acc[c]))
        print('fidelity {:.0f}%'.format(100. * per_class_fidelity[c]))
        print('both_correct_and_same {:.0f}%'.format(100. * per_class_both_correct_and_same[c]))

    print('\nTarget Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        target_test_loss, target_correct, len(train_dataloader.dataset),
        100. * target_correct / len(train_dataloader.dataset)))

    print('\nPseudo Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        pseudo_test_loss, pseudo_correct, len(train_dataloader.dataset),
        100. * pseudo_correct / len(train_dataloader.dataset)))

    print('\nFidelity: {:.0f}%, Both_correct_and_same: {:.0f}%'.format(
        100. * fidelity / len(train_dataloader.dataset), 100. * both_correct_and_same / len(train_dataloader.dataset)))

    return {
        'target_loss': target_test_loss,
        'target_correct': 100. * target_correct / len(train_dataloader.dataset),
        'pseudo_loss': pseudo_test_loss,
        'pseudo_correct': 100. * pseudo_correct / len(train_dataloader.dataset),
        'fidelity': 100. * fidelity / len(train_dataloader.dataset),
        'class_total': class_total,
        'per_class_target_acc': per_class_target_acc,
        'per_class_pseudo_acc':per_class_pseudo_acc,
        'per_class_fidelity': per_class_fidelity,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }




class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """
    @acknowledgment:This code is based on the a publicly available code repository.<https://github.com/thuml/Transfer-Learning-Library>
    @author: Junguang Jiang
    """
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\

    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::

        >>> from tllib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        # print(self.index_matrix.shape)
        # exit()


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        # print(self.kernels[0](features).shape)
        # exit()

        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix
class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))




def writer(train_result, test_result, train_rec):
    csv_file = 'eopch.csv'
    class_total =train_result['class_total']
    per_class_target_acc = train_result['per_class_target_acc']
    per_class_pseudo_acc =train_result['per_class_pseudo_acc']
    per_class_fidelity =train_result['per_class_fidelity']
    precision =train_result['precision']
    recall =train_result['recall']
    f1 =train_result['f1']
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Target Train set', 'Average loss', 'Accuracy'])
        writer.writerow([123, round(train_result['target_loss'], 4), train_result['target_correct']])
        writer.writerow(['Pseudo Train set', 'Average loss', 'Accuracy', 'Fidelity'])
        writer.writerow([123, round(train_result['pseudo_loss'], 4), train_result['pseudo_correct'], train_result['fidelity']])
        writer.writerow(['Class', 'total', 'target_acc', 'pseudo_acc', 'pre', 'rec', 'f1'])
        for i in range(10):
            writer.writerow([i, class_total[i], per_class_target_acc[i], per_class_pseudo_acc[i], precision[i], recall[i], f1[i]])
        writer.writerow(['Target ssim', 'Pseudo ssim', 'Target psnr', 'Pseudo psnr', 'Target mse', 'Pseudo mse', 'Target_pseudo_mse'])
        writer.writerow([train_rec['target_ssim'], train_rec['pseudo_ssim'], train_rec['target_psnr'], train_rec['pseudo_psnr'], train_rec['target_loss'], train_rec['pseudo_loss'], train_rec['target_pseudo_mse']])
        writer.writerow(['Target Test set', 'Average loss', 'Accuracy'])
        writer.writerow([123,round(test_result['target_loss'], 4), test_result['target_correct']])
        writer.writerow(['Pseudo Test set', 'Average loss', 'Accuracy', 'Fidelity'])
        writer.writerow([123,round(test_result['pseudo_loss'], 4), test_result['pseudo_correct'], test_result['fidelity']])

@torch.no_grad()
def private_distribution_em(
    pseudo_top_teacher,
    target_splitnn,
    target_loader,
    aux_dist,
    device,
    num_classes=10,
    temperature=1.0,
    max_iter=10,
    tol=1e-3,
    prior_smoothing=0.01,
    init_prior=None,
    eps=1e-8
):
    pseudo_top_teacher.eval()
    target_splitnn.eval()

    aux_dist_t = torch.as_tensor(aux_dist, device=device, dtype=torch.float32)
    aux_dist_t = aux_dist_t + eps
    aux_dist_t = aux_dist_t / aux_dist_t.sum()

    all_probs = []

    for target_data, _ in target_loader:
        target_data = target_data.to(device)

        _ = target_splitnn(target_data)
        z_t = target_splitnn.intermidiate_to_top.detach()

        teacher_logits = pseudo_top_teacher(z_t)

        # 先不要锐化。temperature=1.0 或 1.5 更稳。
        probs = torch.softmax(teacher_logits / temperature, dim=1)

        all_probs.append(probs)

    probs = torch.cat(all_probs, dim=0)  # [N, C]

    soft_init = probs.mean(dim=0)
    soft_init = soft_init + eps
    soft_init = soft_init / soft_init.sum()

    hard_pred = probs.argmax(dim=1)
    hard_count = torch.bincount(hard_pred, minlength=num_classes).float().to(device)
    hard_count = hard_count + eps
    hard_count = hard_count / hard_count.sum()

    if init_prior is None:
        pi = soft_init.clone()
    else:
        pi = init_prior.detach().clone().to(device).float()
        pi = pi + eps
        pi = pi / pi.sum()

    trace = [pi.detach().clone()]

    for _ in range(max_iter):
        pi_old = pi.clone()

        # label-shift correction weight
        weight = pi / aux_dist_t

        # E-step
        q = probs * weight.unsqueeze(0)
        q = q / (q.sum(dim=1, keepdim=True) + eps)

        # M-step
        pi = q.mean(dim=0)

        # smoothing，避免类别被过早压到 0
        if prior_smoothing > 0:
            uniform = torch.ones_like(pi) / num_classes
            pi = (1.0 - prior_smoothing) * pi + prior_smoothing * uniform

        pi = pi + eps
        pi = pi / pi.sum()

        trace.append(pi.detach().clone())

        if torch.norm(pi - pi_old, p=1).item() < tol:
            break

    return pi


@torch.no_grad()
def estimate_private_distribution_epoch(
    pseudo_top_teacher,
    target_splitnn,
    target_loader,
    private_distributed,
    aux_dist,
    device,
    alpha=0.0,          # 建议先用 0.0；若你坚持用校准后分布来估 prior，再改 > 0
    use_calibrated=False
):
    pseudo_top_teacher.eval()
    target_splitnn.eval()

    num_classes = len(aux_dist)
    epoch_num = torch.zeros(num_classes, device=device)
    epoch_den = torch.tensor(0.0, device=device)

    aux_dist_t = torch.as_tensor(aux_dist, device=device, dtype=torch.float32)
    log_prior_shift = torch.log(private_distributed + 1e-8) - torch.log(aux_dist_t + 1e-8)
    log_prior_shift = log_prior_shift.clamp(-2.0, 2.0)

    for target_data, _ in target_loader:
        target_data = target_data.to(device)

        _ = target_splitnn(target_data)
        target_intermidiate_to_top = target_splitnn.intermidiate_to_top.detach()

        teacher_logits = pseudo_top_teacher(target_intermidiate_to_top)

        if use_calibrated:
            # 不太推荐，但如果你想保持和当前逻辑接近，可以这么做
            probs = torch.softmax(teacher_logits + alpha * log_prior_shift.unsqueeze(0), dim=1)
        else:
            # 更推荐：用 raw teacher probs 估 prior，减少自反馈
            probs = torch.softmax(teacher_logits / 0.5, dim=1)

        conf, _ = torch.max(probs, dim=1)
        '''entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)


        epoch_num += (probs * sample_w.unsqueeze(1)).sum(dim=0)
        epoch_den += sample_w.sum()'''



        epoch_num += probs.sum(dim=0)

    epoch_dist = epoch_num / epoch_num.sum()

    return epoch_dist




def compute_mmd_alignment(source_feat, target_feat, sigmas=(1, 2, 4, 8, 16)):
    """轻量 multi-kernel MMD，仅在 GAP 后的通道向量上对齐源 / 目标分布。
    用法：与对抗对齐互补，不替换。
    """
    if source_feat.dim() == 4:
        source_feat = source_feat.mean(dim=(2, 3))
    if target_feat.dim() == 4:
        target_feat = target_feat.mean(dim=(2, 3))

    Bs = source_feat.size(0)
    Bt = target_feat.size(0)
    if Bs == 0 or Bt == 0:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)

    xx = source_feat
    yy = target_feat
    # pairwise squared L2
    dxx = torch.cdist(xx, xx, p=2).pow(2)
    dyy = torch.cdist(yy, yy, p=2).pow(2)
    dxy = torch.cdist(xx, yy, p=2).pow(2)

    loss = torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s ** 2) + 1e-8)
        kxx = torch.exp(-gamma * dxx).mean()
        kyy = torch.exp(-gamma * dyy).mean()
        kxy = torch.exp(-gamma * dxy).mean()
        loss = loss + (kxx + kyy - 2.0 * kxy)
    return loss / len(sigmas)


def compute_ldfa_infonce(
    source_feat,          # [Bs, D] or [Bs, C, H, W]
    target_feat,          # [Bt, D] or [Bt, C, H, W]
    source_label,         # [Bs]      shadow 真实标签
    calibrated_labels,    # [Bt]      teacher 校准后伪标签
    target_weight,        # [Bt]      teacher 置信度（calibrated_conf）
    num_classes=10,
    temperature=0.1,
    conf_threshold=0.7,
    min_per_class=2,
    eps=1e-8,
):
    """
    Class-conditional InfoNCE alignment:
      每个 shadow 样本 → 拉向其类 target prototype + 推离其他类 target prototype
    """
    device = source_feat.device

    # 如果是 4D 卷积特征图，先做 GAP 变为 2D 表征
    if source_feat.dim() == 4:
        source_feat = source_feat.mean(dim=(2, 3))
    if target_feat.dim() == 4:
        target_feat = target_feat.mean(dim=(2, 3))

    # ---------- 1. 建立 per-class target prototype（高 conf 才参与） ----------
    proto_list = []
    valid_classes = []
    for c in range(num_classes):
        mask_c = (calibrated_labels == c) & (target_weight > conf_threshold)
        n_c = mask_c.sum().item()
        if n_c >= min_per_class:
            # 按 conf 加权平均比简单 mean 更稳
            w = target_weight[mask_c].unsqueeze(1)  # [n_c, 1]
            proto = (target_feat[mask_c] * w).sum(dim=0) / (w.sum() + eps)
            proto_list.append(proto)
            valid_classes.append(c)

    # 少于 2 类无法 contrast
    if len(valid_classes) < 2:
        return torch.tensor(0.0, device=device, dtype=source_feat.dtype)

    proto_mat = torch.stack(proto_list, dim=0)              # [C', D]
    class_to_idx = {c: i for i, c in enumerate(valid_classes)}

    # ---------- 2. 筛出 shadow 端有对应 target prototype 的样本 ----------
    valid_idx = [class_to_idx.get(int(c), -1) for c in source_label.tolist()]
    valid_idx_t = torch.tensor(valid_idx, device=device)
    keep_mask = valid_idx_t >= 0
    if keep_mask.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=source_feat.dtype)

    src = source_feat[keep_mask]                              # [Bs', D]
    tgt_idx = valid_idx_t[keep_mask]                         # [Bs'] — 正类 prototype 索引

    # ---------- 3. L2 归一化 + 余弦相似度 ----------
    src_n   = F.normalize(src, dim=1)
    proto_n = F.normalize(proto_mat, dim=1)
    logits = src_n @ proto_n.t() / temperature                # [Bs', C']

    # ---------- 4. InfoNCE = cross-entropy over prototypes ----------
    loss = F.cross_entropy(logits, tgt_idx)
    return loss