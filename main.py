import argparse
import os
import random
from collections import Counter
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_1
import copy
import torchvision.utils as vutils

from labelshit import build_two_disjoint_label_shift_subsets, show_class_distribution, label_shift_subsets
# from model_ import Discriminator, resnet18,AE_Model,CINIC_Inversion, vgg16
from model import cifar_decoder, cifar_discriminator_model, vgg16, cifar_mobilenet, vgg16
from torch.utils.tensorboard import SummaryWriter
from splitnn import Client, Server, SplitNN, Top
from utils import attack_test, pseudo_training, cla_pseudo_test, attack_cla_train, writer, Pre_training, \
    estimate_private_distribution_epoch, private_distribution_em
import numpy as np
from torch.utils.data import Subset
from random import shuffle
import math
from utils import MultipleKernelMaximumMeanDiscrepancy,GaussianKernel



def main():
    parser = argparse.ArgumentParser(description="Demo of FORA in CIFAR10")
    parser.add_argument('--iteration', type=int, default=10000, help="")
    parser.add_argument('--lr', type=float, default=1e-4, help="")
    parser.add_argument('--server_pseudo_lr', type=float, default=1e-4, help="")
    parser.add_argument('--dlr', type=float, default=1e-5, help="")
    parser.add_argument('--save_path', type=str, default='attack_model.pth', help="")
    parser.add_argument('--gid', type=str, default='0', help="gpu id")
    parser.add_argument('--bottom_layer_id', type=int, default='3', help="layer id")
    parser.add_argument('--top_layer_id', type=int, default='3', help="layer id")
    parser.add_argument('--a', type=float, default=1, help='hyperparameter of loss')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--dataset', type=str, default='cifar10', help='')
    parser.add_argument('--pretrain_iter', type=int, default=5000, help='pretrain iterations')
    parser.add_argument('--mmd_weight', type=float, default=0.0, help='MK-MMD alignment weight (0 to disable; ablation showed net-zero gain at b3t3)')
    parser.add_argument('--seed', type=int, default=2025, help='random seed; pass different value per repeat to get variance')
    # === 对比学习 (class-conditional InfoNCE) ===
    parser.add_argument('--ldfa_w', type=float, default=1.0, help='weight of class-conditional InfoNCE on pseudo_bottom (kept at inert/non-harmful setting; forcing it via low temp destabilizes pseudo_bottom)')
    parser.add_argument('--ldfa_temp', type=float, default=0.5, help='InfoNCE temperature on pseudo_bottom; 0.5 is the safe inert value (0.1 sends noisy gradients into pseudo_bottom and HURTS, confirmed b3t3)')
    parser.add_argument('--ldfa_conf', type=float, default=0.4, help='confidence threshold for target pseudo-label prototypes in InfoNCE')
    # === 对比/原型思想的有效落点：EMA 类原型精修蒸馏伪标签（直接作用于分类指标，不扰动对抗对齐） ===
    parser.add_argument('--proto_refine_w', type=float, default=0.3, help='blend weight of EMA-prototype pseudo-labels into the distillation target (0 disables). This is where the class-conditional contrastive idea actually bites label inference.')
    parser.add_argument('--proto_ema', type=float, default=0.9, help='EMA momentum for class prototypes in target-feature space')
    parser.add_argument('--proto_temp', type=float, default=0.1, help='temperature for prototype (cosine) soft assignment')
    parser.add_argument('--proto_conf', type=float, default=0.7, help='confidence threshold for samples that update the prototypes')
    parser.add_argument('--proto_warmup', type=int, default=2000, help='iterations before prototype refinement engages (let prototypes stabilize first)')
    # === label-shift 自适应门控：无偏移时自动关闭校准/先验，修复均匀分布下变差的问题 ===
    parser.add_argument('--shift_adaptive', type=int, default=1, help='1: gate calibration & prior loss by estimated shift strength (self-disables under uniform/no-shift); 0: legacy fixed behavior')
    parser.add_argument('--shift_gate_s0', type=float, default=0.15, help='TV-distance scale at which shift gate saturates to 1.0; below it the correction is suppressed (kills EM noise under no-shift)')
    parser.add_argument('--cal_coef', type=float, default=0.5, help='base coefficient for prior-shift logit calibration (scaled by the adaptive gate)')
    parser.add_argument('--private_dist', type=str, default='shift', choices=['shift', 'uniform'], help="private label distribution: 'shift' (heavy label shift) or 'uniform'")
    parser.add_argument('--dataset_num', type=int, default=10000, help='size of auxiliary data')  # 25000 -> 30000
    parser.add_argument('--dist_id', type=int, default=None, help='0..10: 选 CLIENT_DISTS 里的私有分布')

    args = parser.parse_args()    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        id = 'cuda:'+args.gid
        device = torch.device(id)
        torch.cuda.set_device(id)
        # cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(device)


    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"[seed] {args.seed}")

    start_time = time.time()
    
    cinic_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])


##########新增

    # ---- 合并 CIFAR10 train(50000) + test(10000) -> 一个 60000 的池(每类 6000)----
    train_part = torchvision.datasets.CIFAR10(root='./data', train=True, transform=cinic_transform, download=True)
    test_part = torchvision.datasets.CIFAR10(root='./data', train=False, transform=cinic_transform, download=True)
    origin_dataset = train_part  # 复用对象(保留 transform)
    origin_dataset.data = np.concatenate([train_part.data, test_part.data], axis=0)  # (60000,32,32,3)
    origin_dataset.targets = list(train_part.targets) + list(test_part.targets)  # len 60000

    #################################### label shift #############
    # 私有(client)= 10000,label-shift;辅助(shadow/server)= args.dataset_num(默认 30000),均匀
    # 两者从同一个 60000 池里【不相交】抽取
    CLIENT_DISTS = [
        [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000],  # d0
        [0.1300, 0.1150, 0.1050, 0.0980, 0.0950, 0.0930, 0.0920, 0.0910, 0.0906, 0.0904],  # d1
        [0.1600, 0.1300, 0.1100, 0.0960, 0.0900, 0.0860, 0.0840, 0.0820, 0.0812, 0.0808],  # d2
        [0.1900, 0.1450, 0.1150, 0.0940, 0.0850, 0.0790, 0.0760, 0.0730, 0.0718, 0.0712],  # d3
        [0.2200, 0.1600, 0.1200, 0.0920, 0.0800, 0.0720, 0.0680, 0.0640, 0.0624, 0.0616],  # d4
        [0.2500, 0.1750, 0.1250, 0.0900, 0.0750, 0.0650, 0.0600, 0.0550, 0.0530, 0.0520],  # d5
        [0.2800, 0.1900, 0.1300, 0.0880, 0.0700, 0.0580, 0.0520, 0.0460, 0.0436, 0.0424],  # d6
        [0.3100, 0.2050, 0.1350, 0.0860, 0.0650, 0.0510, 0.0440, 0.0370, 0.0342, 0.0328],  # d7
        [0.3400, 0.2200, 0.1400, 0.0840, 0.0600, 0.0440, 0.0360, 0.0280, 0.0248, 0.0232],  # d8
        [0.3700, 0.2350, 0.1450, 0.0820, 0.0550, 0.0370, 0.0280, 0.0190, 0.0154, 0.0136],  # d9
        [0.4000, 0.2500, 0.1500, 0.0800, 0.0500, 0.0300, 0.0200, 0.0100, 0.0060, 0.0040],  # d10
    ]

    # 私有(client)= 10000,label-shift;辅助(shadow/server)= args.dataset_num(默认 30000),均匀
    # 两者从同一个 60000 池里【不相交】抽取
    if args.private_dist == 'uniform':
        private_dist = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    else:  # 'shift' (default): 重度 label shift
        # private_dist = [0.33, 0.20, 0.15, 0.1, 0.1, 0.04, 0.03, 0.02, 0.02, 0.01]
        # private_dist = [0.30, 0.23, 0.15, 0.1, 0.1, 0.04, 0.03, 0.02, 0.02, 0.01]
        private_dist = CLIENT_DISTS[args.dist_id] if args.dist_id is not None else [0.30, 0.23, 0.15, 0.1, 0.1, 0.04, 0.03, 0.02, 0.02, 0.01]


    print(f"[private_dist={args.private_dist}] {private_dist}")

    aux_dist = [0.1] * 10
    #aux_dist = [0.05, 0.05, 0.1, 0.1, 0.05, 0.2, 0.15, 0.1, 0.1, 0.1]

    train_dataset, shadow_dataset = build_two_disjoint_label_shift_subsets(
        dataset=origin_dataset,
        dist_a=private_dist,
        size_a=10000,
        dist_b=aux_dist,
        size_b=args.dataset_num,
        seed=42
    )

    ###########
    show_class_distribution(train_dataset, origin_dataset)  # 注意:第二个参数改成 origin_dataset(合并后的)
    show_class_distribution(shadow_dataset, origin_dataset)

    print("DataSet:", args.dataset)
    print("Train (private) Dataset:", len(train_dataset))  # 10000
    print("Shadow (aux) Dataset:", len(shadow_dataset))  # args.dataset_num (30000)

    test_dataset = test_part



#################################### label shitf #############

    #[0.15, 0.14, 0.12, 0.11, 0.10, 0.10, 0.09, 0.08, 0.06, 0.05]
    #重度[0.35, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
    # 部分类别缺失[0.30, 0.25, 0.20, 0.15, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00]



    # aux_dist = [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.08, 0.15, 0.25, 0.35]

###########
    print("DataSet:",args.dataset)
    print("Train Dataset:",len(train_dataset))  #50000
    print("Shadow Dataset:", len(shadow_dataset))
    print("Test Dataset:",len(test_dataset))    #10000

    train_dataloader = torch.utils.data.DataLoader(train_dataset,persistent_workers=True, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, persistent_workers=True, batch_size=args.batch_size, shuffle=False, num_workers = 4, pin_memory = True)
    shadow_dataloader = torch.utils.data.DataLoader(shadow_dataset,persistent_workers=True, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True)
#persistent_workers=True

#######################  真实的拆分网络结构 ######################
    target_bottom, target_server, target_top = cifar_mobilenet(bottom_level=args.bottom_layer_id, top_level=args.top_layer_id)
    target_bottom, target_server, target_top = target_bottom.to(device), target_server.to(device), target_top.to(device)

    dataset_shape = train_dataset[0][0].shape   # 3 * 32 * 32

################### 攻击者网络，定义为相同结构 #############################
    #pseudo_bottom, _ , pseudo_top = vgg16(level=int(args.bottom_layer_id), batch_norm=True)
    pseudo_bottom, _, pseudo_top = cifar_mobilenet(bottom_level=args.bottom_layer_id, top_level=args.top_layer_id)
    _, _, pseudo_top_teacher = cifar_mobilenet(bottom_level=args.bottom_layer_id, top_level=args.top_layer_id)
    #_,_ , pseudo_top_teacher = vgg16(level=int(args.bottom_layer_id), batch_norm=True)



    pseudo_bottom, pseudo_top = pseudo_bottom.to(device), pseudo_top.to(device)
    pseudo_top_teacher = pseudo_top_teacher.to(device)
    test_data = torch.ones(1,dataset_shape[0], dataset_shape[1], dataset_shape[2]).to(device)
    with torch.no_grad():
        test_data_output = pseudo_bottom(test_data)
        discriminator_input_shape = test_data_output.shape[1:]
    discriminator = cifar_discriminator_model(discriminator_input_shape, level=args.bottom_layer_id)
    discriminator = discriminator.to(device)

    target_invmodel = cifar_decoder(discriminator_input_shape, args.bottom_layer_id,dataset_shape[0])
    pseudo_invmodel = cifar_decoder(discriminator_input_shape, args.bottom_layer_id,dataset_shape[0])
    target_invmodel, pseudo_invmodel = target_invmodel.to(device), pseudo_invmodel.to(device)


    target_client = Client(target_bottom)
    target_server = Server(target_server)
    target_top = Top(target_top)

    target_bottom_optimizer = optim.Adam(target_bottom.parameters(), lr=1e-3)
    target_server_optimizer = optim.Adam(target_server.parameters(), lr=1e-3)
    target_top_optimizer = optim.Adam(target_top.parameters(), lr=1e-3)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.dlr)
    pseudo_bottom_optimizer = optim.Adam(pseudo_bottom.parameters(),lr=args.lr)
    pseudo_top_optimizer = optim.Adam(pseudo_top.parameters(),lr=args.lr)
    teacher_optimizer  = optim.Adam(pseudo_top_teacher.parameters(),lr=args.lr)
    target_invmodel_optimizer = optim.Adam(target_invmodel.parameters(), lr=1e-3)
    pseudo_invmodel_optimizer = optim.Adam(pseudo_invmodel.parameters(), lr=args.lr)

    target_splitnn = SplitNN(target_client, target_server, target_top, target_bottom_optimizer, target_server_optimizer, target_top_optimizer)

    target_iterator = iter(train_dataloader)
    shadow_iterator = iter(shadow_dataloader)


    private_distributed = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(device)
    # private_distributed = torch.tensor([0.33, 0.20, 0.15, 0.1, 0.1, 0.04, 0.03, 0.02, 0.02, 0.01]).to(device)



######################################### 预训练  ##################################################

    for n in range(1, args.pretrain_iter):
        if (n-1)%int((len(train_dataset)/args.batch_size)) == 0 :
            target_iterator = iter(train_dataloader)
        if (n-1)%int((len(shadow_dataset)/args.batch_size)) == 0 :
            shadow_iterator = iter(shadow_dataloader)
        try:
            target_data, target_label = next(target_iterator)
        except StopIteration:
            target_iterator = iter(train_dataloader)
            target_data, target_label = next(target_iterator)

        try:
            shadow_data, shadow_label = next(shadow_iterator)
        except StopIteration:
            shadow_iterator = iter(shadow_dataloader)
            shadow_data, shadow_label = next(shadow_iterator)

        if target_data.size(0) != shadow_data.size(0):
            print("The number is not match")
            exit()


        target_splitnn_intermidiate, target_intermidiate_to_top = Pre_training(pseudo_top_teacher,teacher_optimizer, target_splitnn, target_invmodel, target_invmodel_optimizer,
                    pseudo_bottom, pseudo_top, pseudo_invmodel, pseudo_invmodel_optimizer,pseudo_bottom_optimizer, pseudo_top_optimizer,
                    discriminator, discriminator_optimizer,
                    target_data, target_label, shadow_data, shadow_label, device, n, args.iteration, args.a)


        if n % int(1000) == 0:
            train_result = attack_cla_train(target_splitnn, pseudo_top, train_dataloader, device, num_classes=10)

######################################### 标签校准 #################################################
    print("预训练结束")

    # Calib 阶段 cosine 学习率衰减——只衰减 pseudo 侧（驱动震荡的来源）。
    # target 已 ~99% train acc，lr 影响小；discriminator 通常需要稳定的对抗强度，不衰减。
    eta_min_ratio = 0.01
    # pseudo_bottom_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     pseudo_bottom_optimizer, T_max=args.iteration, eta_min=args.lr * eta_min_ratio)
    pseudo_top_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        pseudo_top_optimizer, T_max=args.iteration, eta_min=args.lr * eta_min_ratio)
    teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        teacher_optimizer, T_max=args.iteration, eta_min=args.lr * eta_min_ratio)

    for n in range(1, args.iteration + 1):
        if (n - 1) % int((len(train_dataset) / args.batch_size)) == 0:
            target_iterator = iter(train_dataloader)
        if (n - 1) % int((len(shadow_dataset) / args.batch_size)) == 0:
            shadow_iterator = iter(shadow_dataloader)

            # 用上一次估计作为热启动，加速收敛
            epoch_dist = private_distribution_em(
                pseudo_top_teacher=pseudo_top_teacher,
                target_splitnn=target_splitnn,
                target_loader=train_dataloader,
                aux_dist=aux_dist,
                device=device,
                num_classes=10,
                temperature=1.0,
                max_iter=20,
                tol=1e-4,
                prior_smoothing=0.01,
                init_prior=private_distributed
            )
            # print(epoch_dist)

            # 自适应动量：前期慢动量快速吸收估计，后期高动量稳定
            # 之前 0.99 太高，导致前 100+ epoch 实际 prior 几乎仍是均匀，校准 loss 等于无效
            if n < 2000:
                momentum_epoch = 0.7
            elif n < 6000:
                momentum_epoch = 0.9
            else:
                momentum_epoch = 0.97
            private_distributed = momentum_epoch * private_distributed + (1.0 - momentum_epoch) * epoch_dist
            private_distributed = private_distributed + 1e-6
            private_distributed = private_distributed / private_distributed.sum()
        try:
            target_data, target_label = next(target_iterator)
        except StopIteration:
            target_iterator = iter(train_dataloader)
            target_data, target_label = next(target_iterator)

        try:
            shadow_data, shadow_label = next(shadow_iterator)
        except StopIteration:
            shadow_iterator = iter(shadow_dataloader)
            shadow_data, shadow_label = next(shadow_iterator)

        if target_data.size(0) != shadow_data.size(0):
            print("The number is not match")
            exit()

        target_splitnn_intermidiate, target_intermidiate_to_top = pseudo_training(pseudo_top_teacher,teacher_optimizer,target_splitnn, target_invmodel,
                                                                                  target_invmodel_optimizer,
                                                                                  pseudo_bottom, pseudo_top,
                                                                                  pseudo_invmodel,
                                                                                  pseudo_invmodel_optimizer,
                                                                                  pseudo_bottom_optimizer,
                                                                                  pseudo_top_optimizer,
                                                                                  discriminator,
                                                                                  discriminator_optimizer,
                                                                                  target_data, target_label,
                                                                                  shadow_data, shadow_label, device, n,
                                                                                  args.iteration, args.a, private_distributed,aux_dist,
                                                                                  mmd_weight=args.mmd_weight,
                                                                                  ldfa_w=args.ldfa_w, ldfa_temp=args.ldfa_temp, ldfa_conf=args.ldfa_conf,
                                                                                  shift_adaptive=bool(args.shift_adaptive), shift_gate_s0=args.shift_gate_s0, cal_coef=args.cal_coef,
                                                                                  proto_refine_w=args.proto_refine_w, proto_ema=args.proto_ema, proto_temp=args.proto_temp,
                                                                                  proto_conf=args.proto_conf, proto_warmup=args.proto_warmup)

        # pseudo_bottom_scheduler.step()
        pseudo_top_scheduler.step()
        teacher_scheduler.step()

        if n % int(1000) == 0:
            print(private_distributed)
            print(f"  [lr] pseudo_bottom={pseudo_bottom_optimizer.param_groups[0]['lr']:.2e}  "
                  f"pseudo_top={pseudo_top_optimizer.param_groups[0]['lr']:.2e}  "
                  f"teacher={teacher_optimizer.param_groups[0]['lr']:.2e}")
            train_result = attack_cla_train(target_splitnn, pseudo_top, train_dataloader, device, num_classes=10)
            test_result = cla_pseudo_test(target_splitnn, pseudo_top, test_dataloader, device)

        if n % int(5000) == 0:
            pseudo_middle_state = {
                'iteration': n,
                'pseudo_model': pseudo_top.state_dict(),
                'target_top': target_top.state_dict(),
                'final_acc': test_result['pseudo_correct'],
            }

            target_middle_state = {
                'iteration': n,
                'bottom_model': pseudo_bottom.state_dict(),
                'top_model': target_top.state_dict(),
                'final_acc': test_result['pseudo_correct'],
            }
            os.makedirs('./model/pseudo/middle', exist_ok=True)
            os.makedirs('./model/target/middle', exist_ok=True)
            torch.save(target_middle_state, os.path.join('./model/target/middle', str(n) + '_' + args.save_path))
            torch.save(pseudo_middle_state, os.path.join('./model/pseudo/middle', str(n) + '_' + args.save_path))

    sum_train_result = attack_cla_train(target_splitnn, pseudo_top, train_dataloader, device, num_classes=10)
    sum_test_result = cla_pseudo_test(target_splitnn, pseudo_top, test_dataloader, device)
    train_rec = attack_test(target_splitnn, target_invmodel, pseudo_invmodel, device, target_bottom, pseudo_bottom,
                train_dataloader, n, args.iteration)

    writer(sum_train_result, sum_test_result, train_rec)
    print(private_distributed)

    end_time = time.time()

    print(f"程序运行时间: {end_time - start_time:.2f} 秒")

    pseudo_final_state = {
                'iteration': n,
                'pseudo_model': pseudo_top.state_dict(),
                'target_top':target_top.state_dict(),
                'final_acc': test_result['pseudo_correct'],
            }

    target_final_state = {
                'iteration': n,
                'bottom_model': pseudo_bottom.state_dict(),
                'top_model': target_top.state_dict(),
                'final_acc': test_result['pseudo_correct'],
            }

    os.makedirs('./model/pseudo/final', exist_ok=True)
    os.makedirs('./model/target/final', exist_ok=True)
    torch.save(target_final_state, os.path.join('./model/target/final',args.save_path))
    torch.save( pseudo_final_state, os.path.join('./model/pseudo/final',args.save_path))

    print("=> Training Complete!\n")

if __name__ == '__main__':
    main()
