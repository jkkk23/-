# hello

这里是Multi-Task Federated Learning for Personalised Deep Neural Networks in Edge Computing 的复现代码，虽然代码开源，但是其配置到本地以及进行一定的测试也需要一定的工作量，最终得到的结果放在result中。

此外，本次使用的训练集为mnist和cifar10数据集
```
#!/usr/bin/env bash
set -ex

# Reproduces results in Fig. 5 (a), plots the results and creates png of plot.
for seed_value in {1..5}
do
    # FL(FedAvg)
    python main.py -dset mnist -alg fedavg -C 0.5 -B 20 -T 200 -E 1 -device gpu -W 200 -seed $seed_value -lr 0.1 -noisy_frac 0.0 -bn_private none

    # MTFL(FedAvg)
    python main.py -dset mnist -alg fedavg -C 0.5 -B 20 -T 200 -E 1 -device gpu -W 200 -seed $seed_value -lr 0.1 -noisy_frac 0.0 -bn_private yb

    # pFedMe
    python main.py -dset mnist -alg pfedme -C 0.5 -B 20 -T 200 -E 1 -device gpu -W 200 -seed $seed_value -lr 0.3 -noisy_frac 0.0 -beta 1.0 -lamda 1.0

    # Per-FedAvg
    python main.py -dset mnist -alg perfedavg -C 0.5 -B 20 -T 200 -E 1 -device gpu -W 200 -seed $seed_value -lr 0.1 -noisy_frac 0.0 -beta 0.1
done


python ./plot.py ../results/fig_5a.png ../results/dset-mnist_alg-fedavg_C-0.5_B-20_T-200_E-1_device-gpu_W-200_lr-0.1_noisy_frac-0.0_bn_private-none.pkl ../results/dset-mnist_alg-fedavg_C-0.5_B-20_T-200_E-1_device-gpu_W-200_lr-0.1_noisy_frac-0.0_bn_private-yb.pkl ../results/dset-mnist_alg-perfedavg_C-0.5_B-20_T-200_E-1_device-gpu_W-200_lr-0.1_noisy_frac-0.0_beta-0.1.pkl ../results/dset-mnist_alg-pfedme_C-0.5_B-20_T-200_E-1_device-gpu_W-200_lr-0.3_noisy_frac-0.0_beta-1.0_lamda-1.0.pkl
```


