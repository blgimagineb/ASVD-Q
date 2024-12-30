# ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models

Paple link [paper](https://arxiv.org/abs/2312.05821).

# Requirement
- python>=3.10
- pip install -r requirements.txt


对proj_k与proj_v的输入维度用input进行scaling，对proj_k的输出维度用query进行scaling

主要修改文件：
act_aware_utils.py
asvd.py
modules/svd_linear.py


Examples:
```
python asvd.py --model_id="your_model_path" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean_q --kv_cache_ratio_target 0.7 --use_cache --compress_kv_cache --dump_huggingface_model
```

