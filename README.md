# HeaPS
Official PyTorch implementation of HeaPS published in Journal of Parallel and Distributed Computing. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0743731525001352?dgcid=coauthor)<br>
>Federated learning enables collaborative model training among numerous clients. However, existing participant/client selection methods fail to fully leverage the advantages of clients with excellent computational or communication capabilities. In this paper, we propose HeaPS, a novel Heterogeneity-aware Participant Selection framework for efficient federated learning. We introduce a finer-grained global selection algorithm to select communication-strong leaders and computation-strong members from candidate clients. The leaders are responsible for communicating with the server to reduce per-round duration, as well as contributing gradients; while the members communicate with the leaders to contribute more gradients obtained from high-utility data to the global model and improve the final model accuracy. Meanwhile, we develop a gradient migration path generation algorithm to match the optimal leader for each member. We also design the client scheduler to facilitate parallel local training of leaders and members based on gradient migration. Experimental results show that, in comparison with state-of-the-art methods, HeaPS achieves a speedup of up to 3.20× in time-to-accuracy performance and improves the final accuracy by up to 3.57%.<br>

We use Intel Xeon CPU containing a clock rate of 3.0 GHz with 32 cores and utilize 8 Nvidia Tesla V100 GPUs to accelerate training.
The OS system is Ubuntu18.04. The driver version is 440.118.02 and CUDA version is 10.2.
For the base settings, K=50 clients are selected to participant in each round of training from 1.3K=60 clients.<br>
# Quick Start
## Installation
```
conda create -n yourname python=3.8
conda activate yourname
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit==10.2
```
Find your own install command in the official website of PyTorch: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).<br>
The versions and installation methods of the following packages can be found in NVIDIA official website. Note that the versions of packages should correspond to each other.<br>
CUDA Toolkit: [https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)<br>
cuDNN: [https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/install-guide/index.html](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/install-guide/index.html)<br>
NCCL: [https://developer.nvidia.com/nccl/nccl-legacy-downloads](https://developer.nvidia.com/nccl/nccl-legacy-downloads)<br>
## Cloning
```
git clone https://github.com/Dora233/HeaPS
```
To compare with the existing works, Oort and PyramidFL, run the following commands to install [Oort](https://github.com/SymbioticLab/Oort).
```
git clone https://github.com/SymbioticLab/Oort
cd Oort
source install.sh
```
## Dataset Preparation
We use three public datasets of varying scales: Google Speech, OpenImage, and StackOverflow. They can be download from the AI benchmark
[FedScale](https://github.com/SymbioticLab/FedScale) .<br>

## Run Simulation
HeaPS can be tested with training ResNet-34 on non-IID Google Speech by runing the following commands to submit the task:
```
cd {root}/HeaPS/training/evals
python manager.py submit configs/speech/conf_heaps.yml
```
All the configuration files are in ".../HeaPS/training/evals/configs/". <br>
Among them, the suffix "_p" represents using Prox, while without "_p" represents using Yogi. <br>
The suffix "_nomember" represents the HeaPS without member clients used in ablation study. <br>
The other variant used in ablation study is HeaPS without the fine-grained utility. To test it, enter ".../HeaPS/heaps/" and use the content in heaps_util.py to replace the content in heaps.py. <br>
The generated log files are in ".../HeaPS/training/evals/logs/". <br>

## Acknowledgements
Thanks to Chenning Li, Xiao Zeng, Mi Zhang, Zhichao Cao for their MobiCom'22 paper [PyramidFL: a fine-grained client selection framework for efficient federated learning](https://dl.acm.org/doi/10.1145/3495243.3517017). The source codes can be found in repo [PyramidFL](https://github.com/liecn/PyramidFL). <br>
 <br>
We also appreciate the help from Fan Lai, Xiangfeng Zhu, Harsha V. Madhyastha, and Mosharaf Chowdhury for their OSDI'21 paper [Oort: Efficient Federated Learning via Guided Participant Selection.](https://www.usenix.org/conference/osdi21/presentation/lai) The source codes can be found in repo [Oort](https://github.com/SymbioticLab/Oort).
