import argparse
import torch
import os
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_utils import evaluate_model
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential, awq_quant_sequential
from binary_search import binary_search_truncation_rank
import numpy as np
from modules.svd_linear import SVDLinear
def remove_scaling_attributes(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name):
            if hasattr(module, 'in_scale_matrix'):
                del module.scaling_diag_matrix
            if hasattr(module, 'out_scale_matrix'):
                del module.scaling_inverse
def matrix_alpha(matrix, alpha):
    # Clean the matrix

    # Print matrix information for debugging
    matrix = torch.where(torch.isnan(matrix), torch.zeros_like(matrix), matrix)
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals[eigvals <= 0] = 1e-6
    eigvals = eigvals ** alpha
    # Construct result matrix
    result_matrix = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    return result_matrix
def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # if "llama" in model_id or "opt" in model_id:
    #     model = model.to_bettertransformer()
    if not args.raw_model:
        # sensitivity calibration
        calib_loader = get_calib_data(
            args.calib_dataset, tokenizer, model_id, args.n_calib_samples, seed=args.seed, use_bos=args.use_bos
        )
        calib_input_distribution(model, calib_loader, args.scaling_method, args.use_cache)
        if args.sensitivity_metric == "ppl":
            sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
        elif args.sensitivity_metric == "stable_rank":
            sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, args.use_cache)

        # search best truncation rank for each layer

        binary_search_truncation_rank(model, sensitivity, calib_loader, args)

        # quantization
        if args.weight_quant != "none":
            if args.weight_quant == "rtn_int8":
                rtn_quant_sequential(model, 8)
            elif args.weight_quant == "rtn_int6":
                rtn_quant_sequential(model, 6)
            elif args.weight_quant == "awq_int8":
                model = awq_quant_sequential(model, tokenizer, 8)
            elif args.weight_quant == "awq_int4":
                model = awq_quant_sequential(model, tokenizer, 4)

    # evaluate
    remove_scaling_attributes(model)
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "mmlu" if args.eval_mmlu else args.eval_tasks,
        eval_ppl=args.eval_ppl,
        limit=-1,
        use_bos=args.use_bos,
    )

    print(result)
    if not os.path.exists("output"):
        os.makedirs("output")
    with open("output/result.txt", "a+") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")
    if args.dump_huggingface_model:
        if args.act_aware:
            save_path = f"{args.model_id.split('/')[-1]}_ratio-{args.kv_cache_ratio_target}_sample-{args.n_calib_samples}_alpha-{args.alpha}-{args.scaling_method}"
        else:
            save_path = f"{args.model_id.split('/')[-1]}_ratio-{args.kv_cache_ratio_target}_sample-{args.n_calib_samples}-svd"
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        config = model.config.to_dict()
        config["truncation_ranks"] = {}
        for name, module in model.named_modules():
            if isinstance(module, SVDLinear):
                config["truncation_ranks"][name] = module.truncation_rank
        if "opt" in model_id:
            config["auto_map"] = {
                "AutoConfig": "configuration_asvd_opt.ASVDOPTConfig",
                "AutoModelForCausalLM": "modeling_asvd_opt.ASVDOPTForCausalLM",
            }
            config["architectures"] = ["ASVDOPTForCausalLM"]
            os.system(
                "cp ./huggingface_repos/configuration_asvd_opt.py ./huggingface_repos/modeling_asvd_opt.py ./"
                + save_path
            )
        elif "Llama" in model_id:
            config["auto_map"] = {
                "AutoConfig": "configuration_asvd_llama.ASVDLlamaConfig",
                "AutoModelForCausalLM": "modeling_asvd_llama.ASVDLlamaForCausalLM",
            }
            config["architectures"] = ["ASVDLlamaForCausalLM"]
            config["rope_scaling"]={
                "factor": 32.0,
                "type": "dynamic"
            }
            os.system(
                "cp ./huggingface_repos/configuration_asvd_llama.py ./huggingface_repos/modeling_asvd_llama.py ./"
                + save_path
            )
        import json

        json.dump(config, open(save_path + "/config.json", "w"), indent=2)



    # finished


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean", "abs_mean_q"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6", "awq_int8", "awq_int4"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--eval_ppl",
        default="wikitext2,ptb",
        type=str,
    )
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed, which can significantly affect the calibration results",
    )
    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--kv_cache_ratio_target",
        type=float,
        default=-1,
        help="kv cache ratio",
    )
    parser.add_argument(
        "--rank_align",
        type=int,
        default=1,
        help="align rank in SVD",
    )
    parser.add_argument(
        "--raw_model",
        action="store_true",
        help="use the raw model without ASVD",
    )
    parser.add_argument(
        "--use_bos",
        action="store_true",
        help="use bos token in calibration",
    )
    parser.add_argument(
        "--dump_huggingface_model",
        action="store_true",
        help="Whether to dump huggingface model or not."
    )
    args = parser.parse_args()

    main(args)
