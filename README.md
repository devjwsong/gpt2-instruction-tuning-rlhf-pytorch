# gpt2-instruction-tuning-rlhf-pytorch
This repository includes the practice codes that fine-tune GPT-2[[1]](#1) for instruction following, using **(1) PPO (Proximal Policy Optimization)**[[2]](#2), which is one of the most famous approaches for RLHF (Reinforcement Learning from Human Feedback), and **(2) DPO (Direct Preference Optimization)**[[3]](#3). This repository uses **UltraFeedback**[[4]](#4), whose details are elaborated in [here](https://huggingface.co/datasets/openbmb/UltraFeedback).

Note that due to the limitations in the size of the model and dataset, it is not guaranteed to achieve comparable performance to that of large-scale open-source/proprietary models. (e.g. GPT-4, LlaMAs, Claudes). The purpose of this repository is to understand the workflow and implementation of PPO and DPO.

<br/>

---

### How to run

#### (1) Install required packages

Make sure to activate your virtual environment and install required package by running:

```shell
pip install -r requirements.txt
```

<br/>

#### (2) Sample / Filter data

You need to load the UltraFeedback dataset, filter out non-English texts, split the train/validation sets for each step (SFT, RM, PPO, and DPO), and form paired response sets (chosen vs rejected) for DPO. Note that we only consider a response with the highest score as chosen and the lowest score as rejected, while discarding the rest of the responses regardless of their scores. This reduces the dataset size, but ensures the difference between two responses for reliability. While the paired responses are not needed for PPO, this project uses the same preference dataset for both PPO and DPO for comparison. For PPO, any responses are not used, but only instructions are used for online generation.

You can change the arguments in `exec_load_dataset.sh` freely. Refer to the description of each argument and its default value [here](#src/load_dataset.py).

```shell
sh exec_load_dataset.sh
```

<br/>

With default arguments, you will see `.data` directory created:

```
.data
  └--sft  # For SFT (Supervised Fine-Tuning)
  └--rm  # For RM (Reward Model)
  └--pref  # For PPO/DPO
```

<br/>

#### (3) Run SFT (Supervised Fine-Tuning)

Train an SFT model starting from the GPT-2 checkpoint on the instruction following datasets.

You can change the arguments in `exec_run_sft.sh` freely. Refer to the description of each argument and its default value [here](#src/run_sft.py).

```shell
sh exec_run_sft.sh
```

<br/>

With default arguments, you will see `.model/sft` directory created:

```
.model
  └--sft
    └--gpt2_sft_epoch=1_loss=x.xxxx
    └--gpt2_sft_epoch=2_loss=x.xxxx
    └--gpt2_sft_epoch=3_loss=x.xxxx
    ...
```

After finishing step (3), you can choose two paths for post-training. For PPO, follow step **(4.a.1)** => **(4.a.2)**. For DPO, follow step **(4.b)**.

<br/>

#### (4.a.1) Run RM (Reward Model) training

Train a reward model starting from one of SFT checkpoints.

You should specify `--sft_model_path` in `exec_run_rm.sh` as one of the checkpoints you trained in step (3). You can change other arguments freely. Refer to the description of each argument and its default value [here](#src/run_rm.py).

```shell
sh exec_run_rm.sh
```

<br/>

With default arguments, you will see `.model/rm` directory created:

```
.model
  └--rm
    └--gpt2_rm_epoch=1_loss=0.xxxx
    └--gpt2_rm_epoch=2_loss=0.xxxx
    ...
```

<br/>

#### (4.a.2) Run PPO (Proximal Policy Optimization)

Post-train one of your SFT checkpoints on the preference samples, using PPO.

You should specify `--sft_model_path` and `--rm_model_path` in `exec_run_ppo.sh` as one of the checkpoints you trained in step (3) and (4.a.1). You can change other arguments freely. Refer to the description of each argument and its default value [here](#src/run_ppo.py).

```shell
sh exec_run_ppo.sh
```

<br/>

With default arguments, you will see `.model/ppo` directory created:

```
.model
  └--ppo
    └--gpt2_ppo_epoch=1_reward=x.xx
    ...

```

<br/>

#### (4.b) Run DPO (Direct Preference Optimization)

Post-train one of your SFT checkpoints on the preference samples, using DPO. Note that DPO does not require any reward model.

You should specify `--sft_model_path` in `exec_run_dpo.sh` as one of the checkpoints you trained in step (3). You can change other arguments freely. Refer to the description of each argument and its default value [here](#src/run_dpo.py).

```shell
sh exec_run_dpo.sh
```

<br/>

With default arguments, you will see `.model/dpo` directory created:

```
.model
  └--dpo
    └--gpt2_dpo_epoch=1_loss=x.xxxx
    ...
```

<br/>

---

### Arguments

#### `src/load_dataset.py`



#### `src/run_sft.py`



#### `src/run_rm.py`



#### `src/run_ppo.py`



#### `src/run_dpo.py`



---

### Reference

<a id="1">[1]</a> Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.([http://www.persagen.com/files/misc/radford2019language.pdf](http://www.persagen.com/files/misc/radford2019language.pdf))

<a id="2">[2]</a> Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

<a id="3">[3]</a> Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *Advances in neural information processing systems*, *36*, 53728-53741.

<a id="4">[4]</a> Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., ... & Sun, M. (2023). Ultrafeedback: Boosting language models with scaled ai feedback. *arXiv preprint arXiv:2310.01377*.



