import torch
import os
import os.path as osp
from pdb import set_trace as st
import time
import numpy as np
import pickle

def compute_diff(trial_weight, weight):
    
    diff = (trial_weight - weight).clone().abs()
    return diff
    
    
def compute_mul(trial_weight, weight):
    diff = (trial_weight - weight).abs()
    mul = (diff * weight).clone().abs()
    return mul
    
    
def compute_div(trial_weight, weight):
    diff = (trial_weight - weight).abs()
    div = (diff / weight).clone().abs()
    return div

    
def compute_trial_weight(trial_weight, weight):
    return trial_weight.clone().abs()

def compute_weight(trial_weight, weight, mean=0):
    return (weight.clone() - mean).abs()

def compute_inv_weight(trial_weight, weight, mean=0):
    return 1 / (weight.clone() - mean).abs()

def filter_layer(name):
    return True
    
class prune_logger():
    def __init__(self, dir):
        if not osp.exists(dir):
            os.makedirs(dir, exist_ok=True)
        path = osp.join(dir, "prune.log")
        self.writer = open(path, "w")
        
    def log(self, info):
        print(info)
        self.writer.write(f"{info}\n")
    
    def close(self):
        self.writer.close()
        
# The ratio and method for embedding and linear are the same
def tailor_model_same_method_ratio(model, nc_weight_coverage, ):
    logger = prune_logger(trial_args.output_dir)
    
    model = model.cpu()


    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(name)
    
    # c = 0
    # for m in model.bert.encoder.layer[0].modules():
    #     if isinstance(m, torch.nn.Linear):
    #         c += 1
    # print(model.bert.encoder.layer[0])
    # print(c)
    # st()
    
    if trial_args.method == "diff":
        compute_score = compute_diff
    elif trial_args.method == "mul":
        compute_score = compute_mul
    elif trial_args.method == "div":
        compute_score = compute_div
    elif trial_args.method == "trial_weight":
        compute_score = compute_trial_weight
    elif trial_args.method == "weight":
        compute_score = compute_weight
    elif trial_args.method == "inv_weight":
        compute_score = compute_inv_weight
    
    fc_cnt = 0
    for (trial_name, trial_module), (name, module) in zip(trial_model.named_modules(), model.named_modules()):
        if (
            (isinstance(module, torch.nn.Linear) and "linear" in trial_args.reinit_part) or
            (isinstance(module, torch.nn.Embedding) and "embedding" in trial_args.reinit_part)
            
        ):
            fc_cnt += 1
            module.score_weight = compute_score(trial_module.weight, module.weight)
            # print(f"{name} weight")
        if (
            (isinstance(module, torch.nn.Linear) and "linear_bias" in trial_args.reinit_part)
        ):
            fc_cnt += 1
            module.score_bias = compute_score(trial_module.bias, module.bias)
            # print(f"{name} bias")
        elif (isinstance(module, torch.nn.LayerNorm) and "norm" in trial_args.reinit_part):
            raise NotImplementedError
            fc_cnt += 1
            if "weight" in trial_args.method:
                module.score = compute_score(trial_module.weight, module.weight, mean=1)
            else:
                module.score = compute_score(trial_module.weight, module.weight)


    
    if "linear" in trial_args.reinit_part:
        # replace_number & fill = -1 for 
        if trial_args.replace_number == -1:
            fill = -1
        elif trial_args.replace_number == 0:
            fill = 0
        logger.log(f"*****Prune Linear Weight*****")
        if trial_args.per_layer_prune:
            raise NotImplementedError
            model = perlayer_score_prune(model, trial_args.ratio, torch.nn.Linear, logger)
        else:
            model = score_prune(model, trial_args.ratio, torch.nn.Linear, logger, fill=fill, attr="weight")
    if "linear_bias" in trial_args.reinit_part:
        logger.log(f"*****Prune Linear Bias*****")
        if trial_args.per_layer_prune:
            raise NotImplementedError
            model = perlayer_score_prune(model, trial_args.ratio, torch.nn.Linear, logger)
        else:
            model = score_prune(model, trial_args.ratio, torch.nn.Linear, logger, attr="bias")
    if "embedding" in trial_args.reinit_part:
        if trial_args.replace_number == -1:
            fill = -1
        elif trial_args.replace_number == 0:
            fill = 0
        logger.log(f"*****Prune Embedding*****")
        if trial_args.per_layer_prune:
            model = perlayer_score_prune(model, trial_args.ratio, torch.nn.Embedding, logger)
        else:
            model = score_prune(model, trial_args.ratio, torch.nn.Embedding, logger, fill=fill)
    if "norm" in trial_args.reinit_part:
        logger.log(f"*****Prune Norm*****")
        if trial_args.per_layer_prune:
            model = perlayer_score_prune(model, trial_args.ratio, torch.nn.LayerNorm, logger, fill=1)
        else:
            model = score_prune(model, trial_args.ratio, torch.nn.LayerNorm, logger, fill=1)
    logger.close()
    
    return model

# The ratio and method for embedding and linear are the same
def tailor_model_two_factors(model, weight_nc_coverage, args):
    logger = prune_logger(args.output_dir)
    
    model = model.cpu()
    weight_nc_names = list(weight_nc_coverage.keys())
    
    total = 0
    for name, module in model.named_modules():
        if ( isinstance(module, torch.nn.Linear) and name in weight_nc_names ):
            total += module.weight.data.numel()
    
    
    total_weights, total_nc = torch.zeros(total), torch.zeros(total)
    index = 0
    for name, module in model.named_modules():
        if ( isinstance(module, torch.nn.Linear) and name in weight_nc_names ):
            size = module.weight.data.numel()
            total_weights[index:(index+size)] = module.weight.data.view(-1).abs().clone()
            index += size
    index = 0
    for name, weight_nc in weight_nc_coverage.items():
        size = weight_nc.size
        total_nc[index:(index+size)] = torch.Tensor(weight_nc.flatten())
        index += size
    
    
    y, i = torch.sort(total_weights)
    weight_thre_index = int(total * args.weight_prune_ratio)
    if weight_thre_index == len(y):
        weight_thred = y[-1] + 0.1
    else:
        weight_thred = y[weight_thre_index]
    log = f"Pruning weight threshold: {weight_thred:.4f}"
    logger.log(log)
    
    
    y, i = torch.sort(total_nc, descending=True)
    nc_thre_index = int(total * args.nc_maintain_ratio)
    if nc_thre_index == len(y):
        nc_thred = y[-1] + 0.1
    else:
        nc_thred = y[nc_thre_index]
    log = f"Pruning NC threshold: {nc_thred:.4f}"
    logger.log(log)
    
    
    pruned = 0
    zero_flag = False
    
    
    for name, module in model.named_modules():
        if ( isinstance(module, torch.nn.Linear) and name in weight_nc_names):
            score_copy = module.weight.data.abs().clone()
            weight_mask = score_copy.gt(weight_thred).float()
            nc_mask = torch.Tensor(weight_nc_coverage[name]).gt(nc_thred).float()
            mask = torch.logical_or(weight_mask, nc_mask)
            
            pruned = pruned + mask.numel() - torch.sum(mask)

            module.weight.data.mul_(mask)
            
            # module.weight.data.fill_(0.0)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t total params: {mask.numel()} \t "
            f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})")
            logger.log(log)
            
    pruned = int(pruned)
    log = (f"Total conv params: {total}, Pruned conv params: {pruned}, "
    f"Pruned ratio: {pruned/total:.2f}")
    logger.log(log)
    
    
    logger.close()
    model = model.cuda()
    
    return model


# The ratio and method for embedding and linear are the same
# Prune low NC and high weight
def tailor_model(model, weight_nc_coverage, nc_args):
    logger = prune_logger(nc_args.output_dir)


    model = model.cpu()
    weight_nc_names = list(weight_nc_coverage.keys())
    
    total = 0
    for name, module in model.named_modules():
        if ( isinstance(module, torch.nn.Linear) and name in weight_nc_names ):
            total += module.weight.data.numel()
    
    
    total_weights, total_nc = np.zeros(total), np.zeros(total)
    index = 0
    for name, module in model.named_modules():
        if ( isinstance(module, torch.nn.Linear) and name in weight_nc_names ):
            size = module.weight.data.numel()
            total_weights[index:(index+size)] = module.weight.data.cpu().view(-1).abs().clone().numpy()
            index += size
    index = 0
    for name, weight_nc in weight_nc_coverage.items():
        size = weight_nc.size
        total_nc[index:(index+size)] = weight_nc.flatten()
        index += size

    ord_nc_path = osp.join(nc_args.nc_profile_dir, "nc_ord.pkl")
    ord_weight_path = osp.join(nc_args.nc_profile_dir, "weight_ord.pkl")
    
    if not osp.exists(ord_nc_path) or True:
        log = f"Computing NC ord score..."
        logger.log(log)
        ord_score_nc = np.argsort(np.argsort(total_nc))
        with open(ord_nc_path, "wb") as f:
            pickle.dump(ord_score_nc, f)
    else:
        with open(ord_nc_path, "rb") as f:
            ord_score_nc = pickle.load(f)

    if not osp.exists(ord_weight_path) or True:
        log = f"Computing Weight ord score..."
        logger.log(log)
        ord_score_weight = np.argsort(np.argsort(total_weights))
        with open(ord_weight_path, "wb") as f:
            pickle.dump(ord_score_weight, f)
    else:
        with open(ord_weight_path, "rb") as f:
            ord_score_weight = pickle.load(f)

    relative_ord_score = ord_score_nc - ord_score_weight

    
    y = np.sort(relative_ord_score, )
    
    thre_index = int(total * nc_args.weight_prune_ratio)
    if thre_index == len(y):
        weight_thred = y[-1] + 0.1
    else:
        weight_thred = y[thre_index]
    log = f"Pruning relative_ord_score threshold: {weight_thred:.4f}"
    logger.log(log)
    
    pruned = 0
    zero_flag = False
    weight_idx = 0
    
    for name, module in model.named_modules():
        if ( isinstance(module, torch.nn.Linear) and name in weight_nc_names):
            score_copy = module.weight.data.abs().clone()
            weight_size = module.weight.numel()
            weight_ord_score = relative_ord_score[weight_idx:weight_idx+weight_size]
            weight_ord_score = weight_ord_score.reshape(score_copy.shape)
            weight_idx += weight_size
            weight_ord_score = torch.Tensor(weight_ord_score)
            mask = weight_ord_score.gt(weight_thred)

            pruned = pruned + mask.numel() - torch.sum(mask)

            module.weight.data.mul_(mask)
            
            # module.weight.data.fill_(0.0)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t total params: {mask.numel()} \t "
            f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})")
            logger.log(log)
            
    pruned = int(pruned)
    log = (f"Total conv params: {total}, Pruned conv params: {pruned}, "
    f"Pruned ratio: {pruned/total:.2f}")
    logger.log(log)
    # exit()
    
    logger.close()
    model = model.cuda()
    
    return model
