<div align="center">

<h1>Sparse Low-rank Adaptation of Pre-trained Language Models</h1>

</div>

🎉  This is the implementation of EMNLP 2023 paper：[Sparse Low-rank Adaptation of Pre-trained Language Models](https://arxiv.org/abs/2311.11696)


## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

## Preparation

### Prepare the Data and Modify the Data Path

In the paper/code, we use the GLUE datasets, you can download the data from Huggingface or from our [Google Drive](https://drive.google.com/drive/folders/1sNoQIp1x-5aXH4r9dOoSdsm5F1kihg_W?usp=sharing)

After download the data, please replace the following data path definition with your data path:

- `main_dir` in Line 27 of `SoRA/src/glue_tasks.py`
- `main_dir` in Line 9 of `SoRA/src/processor.py`
- `data_path` in Line 88 of `SoRA/run_glue.py`, `SoRA/run_glue_adapter.py`, `SoRA/run_glue_bitfit.py` and `SoRA/run_glue_adapter.py`

### Prepare the model

You can download the base model and the corresponding tokenizer from Huggingface. And after that, do not forget to modify the `model_name_or_path` and `tokenizer_name` in script file (.sh).


## Baseline

We provide the implementation of LoRA, Adapter, BitFit and Full-parameter Fine-Tune. You can apply these baselines by running the following codes:

```bash
cd scripts
# LoRA
bash run_glue_lora.sh
# Adapter
bash run_glue_adapter.sh
# BitFit
bash run_glue_bitfit.sh
# Full-parameter Fine-Tune
bash run_glue_finetune.sh
```

## SoRA

You can apply SoRA by running the following codes:

```bash
cd scripts
# without the sparsifying scheduler
bash run_glue_sora_no_schedule.sh
# with the sparsifying scheduler (Algorithm 1)
bash run_glue_sora_schedule_dense.sh
```

We explain some of the arguments as follows:

- `sparse_lambda`: The hyperparameters $\eta_t$ in paper.
- `sparse_lambda_2`: The hyperparameters $\xi$ in paper.
- `lora_r`: The hyperparameters $r_{max}$ in paper.
- `train_sparse`: The argument to decide whether or not to apply SoRA.
- `lambda_schedule`: The strategies for sparsifying schedulers. Possible values are `linear`, `log_linear` and `exp_linear`.
- `max_lambda`: The max $\xi$ when applying sparsifying scheduler.
- `lambda_num`: The num of the indicator $\xi$ when applying sparsifying scheduler.


## Bugs or questions?

If you have any questions related to the codes or the paper, please contact Ning Ding (`dn97@mail.tsinghua.edu.cn`), Xingtai Lv (`lvxt20@mails.tsinghua.edu.cn`) or open an issue.

## Citation

If you find our work useful, please use the following citation: 

```bibtex
@article{ding2023sparse,
  title={Sparse Low-rank Adaptation of Pre-trained Language Models},
  author={Ding, Ning and Lv, Xingtai and Wang, Qiaosen and Chen, Yulin and Zhou, Bowen and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2311.11696},
  year={2023}
}
```
