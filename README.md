# HeaPS
Official PyTorch implementation of "HeaPS: Heterogeneity-aware Participant Selection for Efficient Federated Learning".<br>
>Federated learning promotes collaborative model training among a large number of edge devices or clients. However, selecting participants from clients with heterogeneous data and system performance poses a challenging task. Existing selection algorithms often focus on addressing one aspect of heterogeneity, such as considering only data heterogeneity or system heterogeneity, while neglecting the other. This results in low system efficiency or statistical efficiency. In this paper, we propose a novel heterogeneity-aware selection algorithm. It introduces a finer-grained client utility to decouple the computational and communication capabilities of clients, and then selects participants based on their advantages. Specifically, it selects leader clients with better communication conditions and member clients with faster training speed and higher data quality. We introduce gradient migration between selected members and leaders and develop a path generating algorithm. The algorithm determines the optimal migration path for each member by maximizing the number of its local training iterations. We also design a new local training schedule based on gradient migration to overlap the local training of members with their leaders. Experimental results demonstrate that our algorithm effectively achieves diversified and balanced participant selection and has significant advantages over existing methods in achieving faster training speed and better model accuracy.<br>

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
The other variant used in ablation study is HeaPS without the fine-grained utility, to test it,  enter ".../HeaPS/heaps/" and use the content in heaps_util.py to replace the content in heaps.py. <br>
The generated log files are in ".../HeaPS/training/evals/logs/". <br>

## Acknowledgements
Thanks to Fan Lai, Xiangfeng Zhu, Harsha V. Madhyastha, and Mosharaf Chowdhury for their OSDI'21 paper [Oort: Efficient Federated Learning via Guided Participant Selection.](https://www.usenix.org/conference/osdi21/presentation/lai) The source codes can be found in repo [Oort](https://github.com/SymbioticLab/Oort). <br>
We also appreciate the help from Chenning Li, Xiao Zeng, Mi Zhang, Zhichao Cao for their MobiCom'22 paper ClusterFL: [PyramidFL: a fine-grained client selection framework for efficient federated learning](https://dl.acm.org/doi/10.1145/3495243.3517017). The source codes can be found in repo [PyramidFL](https://github.com/liecn/PyramidFL).
