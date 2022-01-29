# PyramidFL

This repository contains scripts and instructions for reproducing the experiments in our MobiCom'22 paper "
PyramidFL: Fine-grained Data and System Heterogeneity-aware Client Selection for Efficient Federated Learning". 
<!-- [PyramidFL: Fine-grained Data and System Heterogeneity-aware Client Selection for Efficient Federated Learning](https://www.usenix.org/conference/osdi21/presentation/lai)".  -->

# Overview

* [Environment Setting](#environment-setting)
* [Repo Structure](#repo-structure)
* [Reproduce Results with Provided Logs](#reproduce-results-with-provided-logs)
* [Reproduce Results from Scratch](#reproduce-results-from-scratch)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

# Environment Setting

Please run ```install.sh``` to install the following automatically:
* Anaconda Package Manager
* CUDA 10.2

Note: if you prefer different versions of conda and CUDA, please check  comments in `install.sh` for details.

Run the following commands to install PyramidFL. 

```
source install.sh 
git clone https://github.com/liecn/PyramidFL.git
cd PyramidFL
```
# Repo Structure

```
Repo Root
+-- oort          # Oort code base.
+-- training
    +-- eval
        +-- configs             # Configuration for tasks
        +-- logs                # Performance logs for tasks
            +--google_speech 
            +--har
            +--openimage
            +--stackoverflow
            +--plot_{}.py       # plotting scripts
        +-- matlab              # Matlab scripts for fig 7-8
        +-- manager.py          # Main file for executing jobs
        +-- {task}_logging_{note}_{date}   # Raw logs
    +-- helper          # Client class and functions
    +-- utils       
    +-- learner.py      # Learner side for emulating clients
    +-- param_server.py # Central parameter server
```

# Reproduce Results with Provided Logs

We provide the raw logs and performance records under the directory `training/eval`. To reproduce the figures in the paper directly, you can run following commands with corresponding python scripts.

``` python
cd training/eval/logs
python plot_perf_openimage_baseline_mobilenet.py.py     #Fig 2(a)
python plot_perf_openimage_baseline_shufflenet.py       #Fig 2(b)
python plot_perf_openimage_baseline_optimized_mobilenet.py          #Fig 6(a)
python plot_perf_openimage_baseline_optimized_shufflenet.py         #Fig 6(b)
python plot_perf_openimage_mobilenet_yogi.py            #Fig 10(a)
python plot_perf_openimage_mobilenet_prox.py            #Fig 10(b)
python plot_perf_openimage_shufflenet_yogi.py           #Fig 10(c)
python plot_perf_openimage_shufflenet_prox.py           #Fig 10(d)
python plot_perf_speech_yogi.py                         #Fig 11(a)
python plot_perf_speech_prox.py                         #Fig 11(b)
python plot_perf_stackoverflow_yogi.py                  #Fig 12(a)
python plot_perf_stackoverflow_prox.py                  #Fig 12(b)
python plot_perf_har_yogi.py                            #Fig 13(a)
python plot_perf_har_prox.py                            #Fig 13(b)

```

``` matlab
cd training/eval/matlab
matlab fig_client.m         #Fig 7(a)(b) and Fig 8(a)(b)
```

Note:
1. Generated figures can be found under current directory (e.g., `logs`, `matlab`).
2. Detailed raw logs can be found under the directory `evals`, according to the date. For example, in script `evals/logs/plot_perf_speech_yogi.py`, we use the performance records `evals/logs/google_speech/0807_041041_28052/aggregator/training_perf`. It corresponds to the raw log `evals/google_speech_logging_random_resnet_yogi_0807_041041_28052`.

# Reproduce Results from Scratch
***Please assure that all paths in the configurations are consistent for datasets, scripts, and logs.***

1. Download datasets. Stackoverflow, Google Speech, and OpenImage can be found on [FedScale](https://github.com/SymbioticLab/FedScale). And HARBox can be found on [FL-Datasets-for-HAR](https://github.com/xmouyang/FL-Datasets-for-HAR)

We show the structure of our dataset directory as follows:
```
Dataset Root
+-- dataset
    +-- google_speech       # Speech recognition
        +-- _background_noise_
        +-- test 
        +-- train
        +-- clientDataMap
    +-- HARBox        # IMU-based human activity recognition
        +-- 0
        +-- ... 
        +-- 120 
    +-- open_images          # Image classification
        +-- test 
        +-- train
        +-- classTags
        +-- vocab_tags.txt
        +-- clientDataMap  
    +-- stackoverflow    # Next-work prediction
        +-- test 
        +-- train
        +-- albert-base-v2-config.json
        +-- vocab_tags.txt
        +-- vocab_tokens.txt  
```
2. Please run following commands to submit the task.
For example, submit the configuration file `evals/configs/speech/conf_random.yml` to the main file `manager.py`.

``` bash
cd {root}/PyramidFL/training/evals
python manager.py submit configs/speech/conf_random.yml
```

3. All logs will be dumped to `log_path` specified in the configuration file. `training_perf` locates at the master node under this path. Meanwhile, the user can check `/evals/{task}_logging_{date}` to see whether the job is moving on.

# Notes
please cite our paper if you think the source codes are useful in your research project.
```bibtex
@inproceedings{PyramidFL_MobiCom22,
    author = {Li, Chenning and Zeng, Xiao and Zhang, Mi and Cao, Zhichao},
    title = {PyramidFL: Fine-grained Data and System Heterogeneity-aware Client Selection for Efficient Federated Learning},
    year = {2022},
    booktitle = {Proceedings of ACM MobiCom},
}
```

# Acknowledgements

Thanks to Fan Lai, Xiangfeng Zhu, Harsha V. Madhyastha, and Mosharaf Chowdhury for their OSDI'21 paper [Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/conference/osdi21/presentation/lai). The source codes can be found in repo [Oort](https://github.com/SymbioticLab/Oort).

We also appreciate the help from Xiaomin Ouyang, Zhiyuan Xie, Jiayu Zhou, Jianwei Huang, and Guoliang Xing for their MobiSys'21 paper [ClusterFL: a similarity-aware federated learning system for human activity recognition](https://dl.acm.org/doi/10.1145/3458864.3467681). The HARBox dataset can be found in repo [FL-Datasets-for-HAR](https://github.com/xmouyang/FL-Datasets-for-HAR).


# Contact
Chenning Li by lichenni@msu.edu


