import torch
import os
import os.path as osp
from pdb import set_trace as st

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
def tailor_model_same_method_ratio(trial_model, model, trial_args):
    logger = prune_logger(trial_args.output_dir)
    
    model = model.cpu()
    trial_model = trial_model.cpu()
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
def tailor_model(trial_model, model, trial_args):
    logger = prune_logger(trial_args.output_dir)
    
    model = model.cpu()
    trial_model = trial_model.cpu()
    
    if "embedding" in trial_args.reinit_part:
        if (not hasattr(trial_args, "embedding_method") or 
            trial_args.embedding_ratio < 0 
            ):
            # assert trial_args.embedding_method is None and trial_args.embedding_ratio < 0
            trial_args.embedding_method = trial_args.method
            trial_args.embedding_ratio = trial_args.ratio
            trial_args.embedding_replace_number = -10
        if trial_args.embedding_replace_number == -10:
            trial_args.embedding_replace_number = trial_args.replace_number
        logger.log(f"*****Embedding method {trial_args.embedding_method}, ratio {trial_args.embedding_ratio:.2f}, replace {trial_args.embedding_replace_number}")
            
        if trial_args.embedding_method == "diff":
            compute_score = compute_diff
        elif trial_args.embedding_method == "mul":
            compute_score = compute_mul
        elif trial_args.embedding_method == "div":
            compute_score = compute_div
        elif trial_args.embedding_method == "trial_weight":
            compute_score = compute_trial_weight
        elif trial_args.embedding_method == "weight":
            compute_score = compute_weight
        elif trial_args.embedding_method == "inv_weight":
            compute_score = compute_inv_weight
        else:
            raise NotImplementedError
    
        fc_cnt = 0
        for (trial_name, trial_module), (name, module) in zip(trial_model.named_modules(), model.named_modules()):
            if (
                isinstance(module, torch.nn.Embedding) 
            ):
                fc_cnt += 1
                module.score_weight = compute_score(trial_module.weight, module.weight)
        
        if trial_args.embedding_replace_number == -1:
            fill = -1
        elif trial_args.embedding_replace_number == 0:
            fill = 0
        logger.log(f"*****Prune Embedding*****")
        if trial_args.per_layer_prune:
            model = perlayer_score_prune(model, trial_args.embedding_ratio, torch.nn.Embedding, logger)
        else:
            model = score_prune(model, trial_args.embedding_ratio, torch.nn.Embedding, logger, fill=fill)
            
            
    if "linear" in trial_args.reinit_part or "linear_bias" in trial_args.reinit_part:
        if (trial_args.linear_method is None or 
            trial_args.linear_ratio < 0 
            ):
            assert trial_args.linear_method is None and trial_args.linear_ratio < 0
            trial_args.linear_method = trial_args.method
            trial_args.linear_ratio = trial_args.ratio
        if trial_args.linear_replace_number == -10:
            trial_args.linear_replace_number = trial_args.replace_number
        logger.log(f"*****Linear method {trial_args.linear_method}, ratio {trial_args.linear_ratio:.2f}, replace {trial_args.linear_replace_number}")
        
        if trial_args.linear_method == "diff":
            compute_score = compute_diff
        elif trial_args.linear_method == "mul":
            compute_score = compute_mul
        elif trial_args.linear_method == "div":
            compute_score = compute_div
        elif trial_args.linear_method == "trial_weight":
            compute_score = compute_trial_weight
        elif trial_args.linear_method == "weight":
            compute_score = compute_weight
        elif trial_args.linear_method == "inv_weight":
            compute_score = compute_inv_weight
        else:
            raise NotImplementedError
            
        fc_cnt = 0
        for (trial_name, trial_module), (name, module) in zip(trial_model.named_modules(), model.named_modules()):
            if (
                isinstance(module, torch.nn.Linear)
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

        if "linear" in trial_args.reinit_part:
            # replace_number & fill = -1 for 
            if trial_args.linear_replace_number == -1:
                fill = -1
            elif trial_args.linear_replace_number == 0:
                fill = 0
            logger.log(f"*****Prune Linear Weight*****")
            if trial_args.per_layer_prune:
                raise NotImplementedError
                model = perlayer_score_prune(model, trial_args.ratio, torch.nn.Linear, logger)
            else:
                model = score_prune(model, trial_args.linear_ratio, torch.nn.Linear, logger, fill=fill, attr="weight")
        if "linear_bias" in trial_args.reinit_part:
            logger.log(f"*****Prune Linear Bias*****")
            if trial_args.per_layer_prune:
                raise NotImplementedError
                model = perlayer_score_prune(model, trial_args.linear_ratio, torch.nn.Linear, logger)
            else:
                model = score_prune(model, trial_args.linear_ratio, torch.nn.Linear, logger, attr="bias")

    logger.close()
    model = model.cuda()
    return model

def score_prune(
    model,
    prune_ratio,
    module_type,
    logger,
    fill=0,
    attr="weight",
):
    score_name = "score_"+attr
    model = model.cpu()
    total = 0
    for name, module in model.named_modules():
        if ( isinstance(module, module_type) and filter_layer(name) ):
            total += getattr(module, attr).data.numel()
    
    total_scores = torch.zeros(total)
    index = 0
    for name, module in model.named_modules():
        if ( isinstance(module, module_type) and filter_layer(name)):
            size = getattr(module, attr).data.numel()
            total_scores[index:(index+size)] = getattr(module, score_name).data.view(-1).abs().clone()
            index += size
    
    
    y, i = torch.sort(total_scores)
    # thre_index = int(total * prune_ratio)
    # thre = y[thre_index]
    thre_index = int(total * prune_ratio)
    if thre_index == len(y):
        thre = y[-1] + 0.1
    else:
        thre = y[thre_index]
    log = f"Pruning threshold: {thre:.4f}"
    logger.log(log)
    
    pruned = 0
    zero_flag = False
    
    
    for name, module in model.named_modules():
        if ( isinstance(module, module_type) and filter_layer(name)):
            score_copy = getattr(module, score_name).data.abs().clone()
            mask = score_copy.gt(thre).float()
            
            pruned = pruned + mask.numel() - torch.sum(mask)
            # np.random.shuffle(mask)
            if fill == 0:
                fill_matrix = torch.zeros(getattr(module, attr).shape)
            elif fill == 1:
                fill_matrix = torch.ones(getattr(module, attr).shape)
            elif fill == -1:
                fill_matrix = torch.zeros(getattr(module, attr).shape)
                fill_matrix.data.normal_(mean=0.0, std=0.02)
            fill_matrix.data.mul_(1-mask)
            
            getattr(module, attr).data.mul_(mask)
            getattr(module, attr).data.add_(fill_matrix)
            
            # module.weight.data.fill_(0.0)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t total params: {mask.numel()} \t "
            f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})")
            logger.log(log)
            
    # if zero_flag:
    #     raise RuntimeError("There exists a layer with 0 parameters left.")
    log = (f"Total conv params: {total}, Pruned conv params: {pruned}, "
    f"Pruned ratio: {pruned/total:.2f}, fill: {fill}")
    logger.log(log)
    # model = model.cuda()
    return model

def perlayer_score_prune(
    model,
    prune_ratio,
    module_type,
    logger,
    fill=0,
):
    model = model.cpu()
    
    # total = 0
    # for name, module in model.named_modules():
    #     if ( isinstance(module, module_type) and filter_layer(name) ):
    #             total += module.weight.data.numel()
    
    # linear_weights = torch.zeros(total)
    # index = 0
    # for name, module in model.named_modules():
    #     if ( isinstance(module, module_type) and filter_layer(name)):
    #         size = module.weight.data.numel()
    #         linear_weights[index:(index+size)] = module.score.data.view(-1).abs().clone()
    #         index += size
    
    # y, i = torch.sort(linear_weights)
    # # thre_index = int(total * prune_ratio)
    # # thre = y[thre_index]
    # thre_index = int(total * prune_ratio)
    # if thre_index == len(y):
    #     thre = y[-1] + 0.1
    # else:
    #     thre = y[thre_index]
    # log = f"Pruning threshold: {thre:.4f}"
    # logger.log(log)

    pruned, all_layer_total = 0, 0
    zero_flag = False
    
    for name, module in model.named_modules():
        if ( isinstance(module, module_type) and filter_layer(name)):
            
            total = module.weight.data.numel()
            all_layer_total += total
            layer_score = module.score.data.abs().view(-1).clone()
            # print(module.weight.data.shape, module.score.data.shape)
            y, i = torch.sort(layer_score)
            # thre_index = int(total * prune_ratio)
            # thre = y[thre_index]
            thre_index = int(total * prune_ratio)
            if thre_index == len(y):
                thre = y[-1] + 0.1
            else:
                thre = y[thre_index]
            
            
            score_copy = module.score.data.abs().clone()
            mask = score_copy.gt(thre).float()
            
            if fill == 0:
                fill_matrix = torch.zeros(module.weight.shape)
            elif fill == 1:
                fill_matrix = torch.ones(module.weight.shape)
            fill_matrix.data.mul_(1-mask)

            pruned = pruned + mask.numel() - torch.sum(mask)
            # np.random.shuffle(mask)
            module.weight.data.mul_(mask)
            module.weight.data.add_(fill_matrix)
            
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t thred: {thre:.2f} \t total params: {mask.numel()} \t "
            f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})")
            logger.log(log)
            
    # if zero_flag:
    #     raise RuntimeError("There exists a layer with 0 parameters left.")
    log = (f"Total conv params: {all_layer_total}, Pruned conv params: {pruned}, "
    f"Pruned ratio: {pruned/all_layer_total:.2f}")
    logger.log(log)
    model = model.cuda()
    
    return model