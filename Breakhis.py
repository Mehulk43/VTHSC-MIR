# VTS (HashNet with ViT Backbone - ICME 2022)
# paper [Vision Transformer Hashing for Image Retrieval, ICME 2022](https://arxiv.org/pdf/2109.12564.pdf)
# HashNet basecode considered from https://github.com/swuxyj/DeepHash-pytorch

from utils.tools_breakhis import *
from network import *
import os
import json
import torch
import torch.optim as optim
import time
import numpy as np
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
import torch.nn.functional as F
import random
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {

        "dataset": "breakhis",
        # "dataset": "covid",

        # "net": AlexNet, "net_print": "AlexNet",
        # "net":ResNet, "net_print": "ResNet",
        # "net": VisionTransformer, "net_print": "ViT-B_32", "model_type": "ViT-B_32", "pretrained_dir": "pretrainedVIT/ViT-B_32.npz",
        "net": VisionTransformer, "net_print": "ViT-B_16", "model_type": "ViT-B_16", "pretrained_dir": "pretrainedVIT/ViT-B_16.npz",

        "bit_list": [64],
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "device": torch.device("cuda:1"), "save_path": "HashNet",
        "epoch": 150, "test_map": 30, "batch_size": 32, "resize_size": 256, "crop_size": 224,
        "info": "HashNet", "alpha": 0.08, "step_continuation": 20,
    }
    config = config_dataset(config)
    return config


def train_val(config, bit):
    start_epoch = 1
    Best_mAP = 0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(
        config)
    config["num_train"] = num_train

    num_classes = config["n_class"]
    hash_bit = bit

    if "ViT" in config["net_print"]:
        vit_config = VIT_CONFIGS[config["model_type"]]
        net = config["net"](vit_config, config["crop_size"], zero_head=True,
                            num_classes=num_classes, hash_bit=hash_bit).to(device)
    else:
        net = config["net"](bit).to(device)

    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"], config["dataset"] + "_" +
                             config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] +
                                "_" + config["net_print"] + "_Bit" + str(bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"], config["dataset"] + "_" +
                                config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + ".txt")
    f = open(results_path, 'a')
    f2 = open(results_path, 'a')

    if os.path.exists(trained_path):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(trained_path)
        net.load_state_dict(checkpoint['net'])
        Best_mAP = checkpoint['Best_mAP']
        start_epoch = checkpoint['epoch'] + 1
    else:
        if "ViT" in config["net_print"]:
            print('==> Loading from pretrained model..')
            net.load_from(np.load(config["pretrained_dir"]))

    optimizer = config["optimizer"]["type"](
        net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = HashNetLoss(config, bit)
    criterion_contrastive = ContrastiveLoss()

    for epoch in range(start_epoch, config["epoch"]+1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        train_loss_contrastive = 0
        train_mixed_loss = 0

        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u = net(image)

            bsz = int((u.size()[0])/2)
            batch=bsz
            if (bsz == 1):
                features = u
                print("batch=1", "features",features)
            
            if bsz >1:
                u1, u2 = torch.split(u, [bsz, bsz], dim=0)

            features = torch.cat([u1.unsqueeze(1), u2.unsqueeze(1)], dim=1)
            if bsz ==1:
                u1 = u
                u2= u 

            loss = criterion(u, label.float(), ind, config)

            loss_contrastive = criterion_contrastive(features, label.float(),batch)

            train_loss += loss.item()
            train_loss_contrastive += loss_contrastive.item()

            mixed_loss =  0.8*loss + 0.2*loss_contrastive 
            train_mixed_loss += mixed_loss.item()

            mixed_loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)
        train_loss_contrastive = train_loss_contrastive / len(train_loader)
        train_mixed_loss = train_mixed_loss / len(train_loader)

        print("\b\b\b\b\b\b\b Mixed loss:%.5f loss:%.5f contloss:%.5f" %
              (train_mixed_loss, train_loss, train_loss_contrastive))
        f.write('Train | Epoch: %d | Mixed loss:%.5f | loss:%.5f | contloss:%.5f\n' % (
            epoch, train_mixed_loss, train_loss, train_loss_contrastive))

        if (epoch) % config["test_map"] == 0:

            tst_binary, tst_label = compute_result(
                test_loader, net, device=device)
            
            trn_binary, trn_label = compute_result(
                dataset_loader, net, device=device)

            _, __,TP,TN,FN,FP = pr_curve_1(trn_binary.numpy(), tst_binary.numpy(
                ), trn_label.numpy(), tst_label.numpy())
            # print("calculating map.......")
            mAP, cum_prec, cum_recall = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                                                   config["topK"])
            index_range = num_dataset // 100
            index = [i * 100 - 1 for i in range(1, index_range + 1)]
            max_index = max(index)
            overflow = num_dataset - index_range * 100
            index = index + [max_index + i for i in range(1, overflow + 1)]
            c_prec = cum_prec[index]
            c_recall = cum_recall[index]
            
            
            pr_data = {
                "index": index,
                "P": c_prec.tolist(),
                "R": c_recall.tolist(),
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN
            }
            
            os.makedirs(os.path.dirname(
                config["pr_curve_path"]), exist_ok=True)
            with open(config["pr_curve_path"], 'w') as f2:
                f2.write(json.dumps(pr_data))
            print("pr curve save to ", config["pr_curve_path"])

            if mAP > Best_mAP:
                Best_mAP = mAP
                if "save_path" in config:
                    save_path = os.path.join(
                        config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
                    os.makedirs(save_path, exist_ok=True)
                    print("save in ", save_path)
                    np.save(os.path.join(save_path, "tst_label.npy"),
                            tst_label.numpy())
                    np.save(os.path.join(save_path, "tst_binary.npy"),
                            tst_binary.numpy())
                    np.save(os.path.join(save_path, "trn_binary.npy"),
                            trn_binary.numpy())
                    np.save(os.path.join(save_path, "trn_label.npy"),
                            trn_label.numpy())
                    torch.save(net.state_dict(), os.path.join(
                        save_path, "model.pt"))
                print(
                    f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
                print(config)

                P, R= pr_curve(trn_binary.numpy(), tst_binary.numpy(
                ), trn_label.numpy(), tst_label.numpy())
                print(f'Precision Recall Curve data:\n"DSH":[{P},{R}],')
                f.write('PR | Epoch %d | ' % (epoch))
                for PR in range(len(P)):
                    f.write('%.5f %.5f ' % (P[PR], R[PR]))
                f.write('\n')

                print("Saving in ", config["save_path"])
                state = {
                    'net': net.state_dict(),
                    'Best_mAP': Best_mAP,
                    'epoch': epoch,
                }
                torch.save(state, best_path)
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
            f.write('Test | Epoch %d | MAP: %.3f | Best MAP: %.3f\n'
                    % (epoch, mAP, Best_mAP))
            print(config)

            state = {
                'net': net.state_dict(),
                'Best_mAP': Best_mAP,
                'epoch': epoch,
            }
            torch.save(state, trained_path)

    f.close()


class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"],
                             bit).float().to(config["device"])
        self.Y = torch.zeros(
            config["num_train"], config["n_class"]).float().to(config["device"])

        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = config["alpha"] * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + \
            dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss


class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels,batch, mask=None):

        device = (torch.device('cuda:1')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]*2

        if labels is not None:
            
            labels = (labels == 1).nonzero().squeeze()

            labels = labels[:, 1]
            
            labels = (labels.reshape(labels.size()[-1], 1)).T

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature.T,contrast_feature),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        if int(labels.shape[0]) == 10:
            mask = mask.repeat(3, 3)
            
            temp2mask=mask[:,:4]
            temp3mask = torch.cat((mask,temp2mask),dim=1)
            temp2mask=temp3mask[:4,:]
            temp3mask = torch.cat((temp3mask,temp2mask),dim=0)

            mask=temp3mask

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )


        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
            

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss=loss.mean()/100

        return loss


if __name__ == "__main__":
    config = get_config()

    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/VIT/HashNet_{config['dataset']}_{bit}.json"
        train_val(config, bit)
