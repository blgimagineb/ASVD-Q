import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss



def GQA_update(input, num_head_out, num_head_in, hidden_size):
    per_head = int(num_head_in / num_head_out)
    head_dim = int(hidden_size / num_head_in)
    matrix = input.view(num_head_out, per_head, head_dim)
    matrix = matrix.mean(dim=1)
    output = matrix.view(-1)
    return output


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    num_key_value_heads = model.config.num_key_value_heads
    num_attention_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    kv_hidden_size = int(hidden_size * num_key_value_heads / num_attention_heads)
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and ("k_proj" in name):
                module.in_scale_matrix = all_scaling_diag_matrix[name].to(module.weight.device)
                if "abs_mean_q" in method:
                    module.out_scale_matrix = all_scaling_diag_matrix[name.replace('k', 'q')].to(module.weight.device)
                else:
                    module.out_scale_matrix = torch.ones(kv_hidden_size).to(module.weight.device)
            if isinstance(module, nn.Linear) and ("v_proj" in name):
                module.in_scale_matrix = all_scaling_diag_matrix[name.replace('v','k')].to(module.weight.device)
                module.out_scale_matrix = torch.ones(kv_hidden_size).to(module.weight.device)

        return
    model.eval()
    # set hook for every Linear layer

    def hook(module, input, output):
        abs_mean_kv = input[0].abs().mean(dim=-2).detach().view(-1)
        module.in_scale_matrix += abs_mean_kv
        abs_mean_q = output[0].abs().mean(dim=-2).detach().view(-1)
        abs_mean_q = GQA_update(abs_mean_q, num_key_value_heads, num_attention_heads, hidden_size)
        module.out_scale_matrix += abs_mean_q


    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("q_proj" in name):
            module.in_scale_matrix = 0
            module.out_scale_matrix = 0
            module.register_forward_hook(hook)
        if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name):
            module.in_scale_matrix = 0
            module.out_scale_matrix = 0

    # get activation distribution
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("q_proj" in name):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name.replace('q','k')] = module.in_scale_matrix
            all_scaling_diag_matrix[name] = module.out_scale_matrix
            del module.in_scale_matrix
            del module.out_scale_matrix
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("k_proj" in name):
            module.in_scale_matrix = all_scaling_diag_matrix[name].to(module.weight.device)
            if "abs_mean_q" in method:
                module.out_scale_matrix = all_scaling_diag_matrix[name.replace('k','q')].to(module.weight.device)
            else:
                module.out_scale_matrix = torch.ones(kv_hidden_size).to(module.weight.device)
        if isinstance(module, nn.Linear) and ("v_proj" in name):
            module.in_scale_matrix = all_scaling_diag_matrix[name.replace('v','k')].to(module.weight.device)
            module.out_scale_matrix = torch.ones(kv_hidden_size).to(module.weight.device)
    torch.save(all_scaling_diag_matrix, cache_file)
