# Scene Graph Benchmark in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

Our paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949) has been accepted by CVPR 2020 (Oral).

## Recent Updates

- [x] 2020.06.23 Add no graph constraint mean Recall@K (ng-mR@K) and no graph constraint Zero-Shot Recall@K (ng-zR@K) [\[link\]](METRICS.md#explanation-of-our-metrics)
- [x] 2020.06.23 Allow scene graph detection (SGDet) on custom images [\[link\]](#SGDet-on-custom-images)
- [x] 2020.07.21 Change scene graph detection output on custom images to json files [\[link\]](#SGDet-on-custom-images)
- [x] 2020.07.21 Visualize detected scene graphs of custom images [\[link\]](#Visualize-Detected-SGs-of-Custom-Images)
- [ ] TODO: Using [Background-Exempted Inference](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/tree/master/lvis1.0#background-exempted-inference) to improve the quality of TDE Scene Graph

## Contents

1. [Overview](#Overview)
2. [Install the Requirements](INSTALL.md)
3. [Prepare the Dataset](DATASET.md)
4. [Metrics and Results for our Toolkit](METRICS.md)
    - [Explanation of R@K, mR@K, zR@K, ng-R@K, ng-mR@K, ng-zR@K, A@K, S2G](METRICS.md#explanation-of-our-metrics)
    - [Output Format](METRICS.md#output-format-of-our-code)
    - [Reported Results](METRICS.md#reported-results)
5. [Faster R-CNN Pre-training](#pretrained-models)
6. [Scene Graph Generation as RoI_Head](#scene-graph-generation-as-RoI_Head)
7. [Training on Scene Graph Generation](#perform-training-on-scene-graph-generation)
8. [Evaluation on Scene Graph Generation](#Evaluation)
9. [**Detect Scene Graphs on Your Custom Images** :star2:](#SGDet-on-custom-images)
10. [**Visualize Detected Scene Graphs of Custom Images** :star2:](#Visualize-Detected-SGs-of-Custom-Images)
11. [Other Options that May Improve the SGG](#other-options-that-may-improve-the-SGG)
12. [Tips and Tricks for TDE on any Unbiased Task](#tips-and-Tricks-for-any-unbiased-taskX-from-biased-training)
13. [Frequently Asked Questions](#frequently-asked-questions)
14. [Citations](#Citations)

## Overview

This project aims to build a new CODEBASE of Scene Graph Generation (SGG), and it is also a Pytorch implementation of the paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949). The previous widely adopted SGG codebase [neural-motifs](https://github.com/rowanz/neural-motifs) is detached from the recent development of Faster/Mask R-CNN. Therefore, I decided to build a scene graph benchmark on top of the well-known [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) project and define relationship prediction as an additional roi_head. By the way, thanks to their elegant framework, this codebase is much more novice-friendly and easier to read/modify for your own projects than previous neural-motifs framework(at least I hope so). It is a pity that when I was working on this project, the [detectron2](https://github.com/facebookresearch/detectron2) had not been released, but I think we can consider [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) as a more stable version with less bugs, hahahaha. I also introduce all the old and new metrics used in SGG, and clarify two common misunderstandings in SGG metrics in [METRICS.md](METRICS.md), which cause abnormal results in some papers.

이 프로젝트는 SGG(Scene Graph Generation)의 새로운 CODEBASE를 구축하는 것을 목표로 하며 [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949) 문서의 Pytorch 구현이기도 합니다. 이전에 널리 채택된 SGG 코드베이스 [neural-motifs](https://github.com/rowanz/neural-motifs)는 Faster/Mask R-CNN의 최근 개발에서 분리되었습니다. 그래서 잘 알려진 [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) 프로젝트 위에 씬 그래프 벤치마크를 구축하고 관계 예측을 추가 roi_head로 정의하기로 했습니다. 그건 그렇고, 우아한 프레임워크 덕분에 이 코드베이스는 이전의 신경 모티프 프레임워크보다 훨씬 초보자에게 친숙하고 자신의 프로젝트에 대해 읽기/수정하기가 더 쉽습니다(적어도 그렇게 되길 바랍니다). 제가 이 프로젝트를 할 당시에 [detectron2](https://github.com/facebookresearch/detectron2)가 출시되지 않은 것이 아쉽지만 [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)는 버그가 적은 보다 안정적인 버전이라고 볼 수 있습니다. 하하하하. 또한 SGG에서 사용되는 모든 기존 및 새로운 메트릭을 소개하고 [METRICS.md](METRICS.md)에서 SGG 메트릭에 대한 두 가지 일반적인 오해를 명확히 합니다. 이는 일부 논문에서 비정상적인 결과를 유발합니다.




### Benefit from the up-to-date Faster R-CNN in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), this codebase achieves new state-of-the-art Recall@k on SGCls & SGGen (by 2020.2.16) through the reimplemented VCTree using two 1080ti GPUs and batch size 8:

Models | SGGen R@20 | SGGen R@50 | SGGen R@100 | SGCls R@20 | SGCls R@50 | SGCls R@100 | PredCls R@20 | PredCls R@50 | PredCls R@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
VCTree | 24.53 | 31.93 | 36.21 | 42.77 | 46.67 | 47.64 | 59.02 | 65.42 | 67.18

<br>

Note that all results of VCTree should be better than what we reported in [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), because we optimized the tree construction network after the publication.

VCTree의 모든 결과는 [Biased Training의 Unbiased Scene Graph Generation](https://arxiv.org/abs/2002.11949)에서 보고한 것보다 더 나을 것입니다. 왜냐하면 출판 후 트리 구성 네트워크를 최적화했기 때문입니다.


### The illustration of the Unbiased SGG from 'Unbiased Scene Graph Generation from Biased Training'

![alt text](demo/teaser_figure.png "from 'Unbiased Scene Graph Generation from Biased Training'")

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Metrics and Results **(IMPORTANT)**
Explanation of metrics in our toolkit and reported results are given in [METRICS.md](METRICS.md)

## Pretrained Models

Since we tested many SGG models in our paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), I won't upload all the pretrained SGG models here. However, you can download the [pretrained Faster R-CNN](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ) we used in the paper, which is the most time consuming step in the whole training process (it took 4 2080ti GPUs). As to the SGG model, you can follow the rest instructions to train your own, which only takes 2 GPUs to train each SGG model. The results should be very close to the reported results given in [METRICS.md](METRICS.md)

[Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949) 논문에서 많은 SGG 모델을 테스트했기 때문에 여기에 사전 훈련된 모든 SGG 모델을 업로드하지 않겠습니다. 그러나 논문에서 사용한  [pretrained Faster R-CNN](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ)을 다운로드할 수 있습니다. 이는 전체 교육 과정에서 가장 시간이 많이 소요되는 단계입니다(4개의 2080ti GPU 사용). SGG 모델과 관련하여 나머지 지침에 따라 각 SGG 모델을 훈련하는 데 2개의 GPU만 필요한 자체 훈련을 수행할 수 있습니다. 결과는 [METRICS.md](METRICS.md)에 제공된 보고된 결과와 매우 유사해야 합니다.

<br>

After you download the [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ), please extract all the files to the directory `/home/username/checkpoints/pretrained_faster_rcnn`. To train your own Faster R-CNN model, please follow the next section.

The above pretrained Faster R-CNN model achives 38.52/26.35/28.14 mAp on VG train/val/test set respectively.

 [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ)을 다운로드한 후 `/home/username/checkpoints/pretrained_faster_rcnn` 디렉토리에 모든 파일의 압축을 풉니다. 자신의 Faster R-CNN 모델을 훈련하려면 다음 섹션을 따르십시오.

위의 사전 훈련된 Faster R-CNN 모델은 VG train/val/test 세트에서 각각 38.52/26.35/28.14 mAp를 달성합니다.

## Faster R-CNN pre-training
The following command can be used to train your own Faster R-CNN model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /home/kaihua/checkpoints/pretrained_faster_rcnn SOLVER.PRE_VAL False
```
where ```CUDA_VISIBLE_DEVICES``` and ```--nproc_per_node``` represent the id of GPUs and number of GPUs you use, ```--config-file``` means the config we use, where you can change other parameters. ```SOLVER.IMS_PER_BATCH``` and ```TEST.IMS_PER_BATCH``` are the training and testing batch size respectively, ```DTYPE "float16"``` enables Automatic Mixed Precision supported by [APEX](https://github.com/NVIDIA/apex), ```SOLVER.MAX_ITER``` is the maximum iteration, ```SOLVER.STEPS``` is the steps where we decay the learning rate, ```SOLVER.VAL_PERIOD``` and ```SOLVER.CHECKPOINT_PERIOD``` are the periods of conducting val and saving checkpoint, ```MODEL.RELATION_ON``` means turning on the relationship head or not (since this is the pretraining phase for Faster R-CNN only, we turn off the relationship head),  ```OUTPUT_DIR``` is the output directory to save checkpoints and log (considering `/home/username/checkpoints/pretrained_faster_rcnn`), ```SOLVER.PRE_VAL``` means whether we conduct validation before training or not.

- ```CUDA_VISIBLE_DEVICES```, ```--nproc_per_node``` : GPU ID와 사용하는 GPU의 수
-  ```--config-file``` : 변경할 수 있는 파라미터들을 가지고 있는 config
- ```SOLVER.IMS_PER_BATCH```, ```TEST.IMS_PER_BATCH``` : 각각 훈련 및 테스트 batch_size 
- ```DTYPE "float16"``` : [APEX](https://github.com/NVIDIA/apex)에서 지원하는 Automatic Mixed Precision를 활성화한다.  
- ```SOLVER.MAX_ITER``` : 최대 iteration 수
- ```SOLVER.STEPS``` : learning rate decay할 step 수
- ```SOLVER.VAL_PERIOD``` : validation 수행 (step?)
- ```SOLVER.CHECKPOINT_PERIOD``` : model checkpoint 저장 (step?) 
- ```MODEL.RELATION_ON``` : relationship head를 끌지 말지 결정(이것은 Faster R-CNN 전용 사전 훈련 단계이므로 관계 헤드를 끔)
- ```OUTPUT_DIR``` : is the output directory to save checkpoints and log (considering `/home/username/checkpoints/pretrained_faster_rcnn`), 
- ```SOLVER.PRE_VAL``` means whether we conduct validation before training or not.


## Scene Graph Generation as RoI_Head

To standardize the SGG, I define scene graph generation as an RoI_Head. Referring to the design of other roi_heads like box_head, I put most of the SGG codes under ```maskrcnn_benchmark/modeling/roi_heads/relation_head``` and their calling sequence is as follows:
![alt text](demo/relation_head.png "structure of relation_head")

SGG를 표준화하기 위해 장면 그래프 생성을 RoI_Head로 정의합니다. box_head와 같은 다른 roi_heads의 디자인을 참조하여 대부분의 SGG 코드를 ```maskrcnn_benchmark/modeling/roi_heads/relation_head```에 넣었으며 호출 순서는 다음과 같습니다:

## Perform training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX``` and ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL``` to select the protocols. 

**세 가지 표준 프로토콜**이 있습니다. 
- (1) 술어 분류(PredCls): 정답 경계 상자와 레이블을 입력으로 사용 
- (2) 장면 그래프 분류(SGCL): 레이블이 없는 정답 경계 상자 사용 
- (3) 장면 그래프 detection(SGDet): 처음부터 SG(scene graph)를 탐지
프로토콜을 선택하기 위해 두 개의 스위치 ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX```와 ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL```를 사용합니다.

For **Predicate Classification (PredCls)**, we need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Predefined Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

우리는 Faster R-CNN 백본과 relation-head feature extractor에 독립적인 파일 ```roi_heads/relation_head/roi_relation_predictors.py```에서 다양한 SGG 모델을 서로 다른 ```relation-head predictors```로 추상화합니다. 사전 정의된 모델을 선택하려면 ```MODEL.ROI_RELATION_HEAD.PREDICTOR```를 사용할 수 있습니다.

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```
For [Iterative-Message-Passing(IMP)](https://arxiv.org/abs/1701.02426) Model (Note that SOLVER.BASE_LR should be changed to 0.001 in SGCls, or the model won't converge):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```
For our predefined Transformer Model (Note that Transformer Model needs to change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).), which is provided by [Jiaxin Shi](https://github.com/shijx12):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```
For [Unbiased-Causal-TDE](https://arxiv.org/abs/2002.11949) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
```

The default settings are under ```configs/e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```

### Customize Your Own Model
If you want to customize your own model, you can refer ```maskrcnn-benchmark/modeling/roi_heads/relation_head/model_XXXXX.py``` and ```maskrcnn-benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py```. You also need to add corresponding nn.Module in ```maskrcnn-benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py```. Sometimes you may also need to change the inputs & outputs of the module through ```maskrcnn-benchmark/modeling/roi_heads/relation_head/relation_head.py```.

자신의 모델을 사용자 정의하려면 ```maskrcnn-benchmark/modeling/roi_heads/relation_head/model_XXXXX.py``` 및 ```maskrcnn-benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py``를 참조하세요. `. 또한 ```maskrcnn-benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py```에 해당 nn.Module을 추가해야 합니다. 때로는 ```maskrcnn-benchmark/modeling/roi_heads/relation_head/relation_head.py```를 통해 모듈의 입력 및 출력을 변경해야 할 수도 있습니다.


### The proposed Causal TDE on [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949)
As to the Unbiased-Causal-TDE, there are some additional parameters you need to know. ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE``` is used to select the causal effect analysis type during inference(test), where "none" is original likelihood, "TDE" is total direct effect, "NIE" is natural indirect effect, "TE" is total effect. ```MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE``` has two choice "sum" or "gate". Since Unbiased Causal TDE Analysis is model-agnostic, we support [Neural-MOTIFS](https://arxiv.org/abs/1711.06640), [VCTree](https://arxiv.org/abs/1812.01880) and [VTransE](https://arxiv.org/abs/1702.08319). ```MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER``` is used to select these models for Unbiased Causal Analysis, which has three choices: motifs, vctree, vtranse.

Note that during training, we always set ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE``` to be 'none', because causal effect analysis is only applicable to the inference/test phase.

Unbiased-Causal-TDE와 관련하여 알아야 할 몇 가지 추가 매개변수가 있습니다. ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE```은 추론(테스트) 중 인과 관계 분석 유형을 선택하는 데 사용됩니다. 여기서 "없음"은 원래 가능성, "TDE"는 총 직접 효과, "NIE"는 자연 간접 효과입니다. 효과, "TE"는 전체 효과입니다. ```MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE```에는 "합계" 또는 "게이트" 두 가지 선택이 있습니다. Unbiased Causal TDE Analysis는 모델에 구애받지 않기 때문에 [Neural-MOTIFS], [VCTree] 및 [VTransE]를 지원합니다. ```MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER```는 모티프, vctree, vtranse의 세 가지 선택이 있는 비편향 인과 분석을 위해 이러한 모델을 선택하는 데 사용됩니다.

인과 관계 분석은 추론/테스트 단계에만 적용할 수 있기 때문에 훈련 중에는 항상 ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE```을 'none'으로 설정합니다.

### Examples of the Training Command
Training Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```
where ```GLOVE_DIR``` is the directory used to save glove initializations, ```MODEL.PRETRAINED_DETECTOR_CKPT``` is the pretrained Faster R-CNN model you want to load, ```OUTPUT_DIR``` is the output directory used to save checkpoints and the log. Since we use the ```WarmupReduceLROnPlateau``` as the learning scheduler for SGG, ```SOLVER.STEPS``` is not required anymore.

Training Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```


## Evaluation

### Examples of the Test Command
Test Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```

Test Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10028 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgcls-exmp OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```

### Examples of Pretrained Causal MOTIFS-SUM models
Examples of Pretrained Causal MOTIFS-SUM models on SGDet/SGCls/PredCls (batch size 12): [(SGDet Download)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs), [(SGCls Download)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781938&authkey=AO_ddcgNpVVGE-g), [(PredCls Download)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781937&authkey=AOzowl5-07RzJz4)

Corresponding Results (The original models used in the paper are lost. These are the fresh ones, so there are some fluctuations on the results. More results can be found in [Reported Results](METRICS.md#reported-results)):

Models |  R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
MOTIFS-SGDet-none   | 25.42 | 32.45 | 37.26 | 4.36 | 5.83 | 7.08 | 0.02 | 0.08 | 0.24
MOTIFS-SGDet-TDE    | 11.92 | 16.56 | 20.15 | 6.58 | 8.94 | 10.99 | 1.54 | 2.33 | 3.03
MOTIFS-SGCls-none   | 36.02 | 39.25 | 40.07 | 6.50 | 8.02 | 8.51 | 1.06 | 2.18 | 3.07
MOTIFS-SGCls-TDE    | 20.47 | 26.31 | 28.79 | 9.80 | 13.21 | 15.06 | 1.91 | 2.95 | 4.10
MOTIFS-PredCls-none | 59.64 | 66.11 | 67.96 | 11.46 | 14.60 | 15.84 | 5.79 | 11.02 | 14.74
MOTIFS-PredCls-TDE  | 33.38 | 45.88 | 51.25 | 17.85 | 24.75 | 28.70 | 8.28 | 14.31 | 18.04

## SGDet on Custom Images
Note that evaluation on custum images is only applicable for SGDet model, because PredCls and SGCls model requires additional ground-truth bounding boxes information. To detect scene graphs into a json file on your own images, you need to turn on the switch TEST.CUSTUM_EVAL and give a folder path that contains the custom images to TEST.CUSTUM_PATH. Only JPG files are allowed. The output will be saved as custom_prediction.json in the given DETECTED_SGG_DIR.

Test Example 1 : (SGDet, **Causal TDE**, MOTIFS Model, SUM Fusion) [(checkpoint)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

Test Example 2 : (SGDet, **Original**, MOTIFS Model, SUM Fusion) [(same checkpoint)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

The output is a json file. For each image, the scene graph information is saved as a dictionary containing bbox(sorted), bbox_labels(sorted), bbox_scores(sorted), rel_pairs(sorted), rel_labels(sorted), rel_scores(sorted), rel_all_scores(sorted), where the last rel_all_scores give all 51 predicates probability for each pair of objects. The dataset information is saved as custom_data_info.json in the same DETECTED_SGG_DIR.

출력은 json 파일입니다. 각 이미지에 대해 장면 그래프 정보는 bbox(sorted), bbox_labels(sorted), bbox_scores(sorted), rel_pairs(sorted), rel_labels(sorted), rel_scores(sorted), rel_all_scores(sorted)를 포함하는 사전으로 저장됩니다. 마지막 rel_all_scores는 각 개체 쌍에 대한 모든 51개 술어 확률을 제공합니다. 데이터 세트 정보는 동일한 DETECTED_SGG_DIR에 custom_data_info.json으로 저장됩니다.

## Visualize Detected SGs of Custom Images
To visualize the detected scene graphs of custom images, you can follow the jupyter note: [visualization/3.visualize_custom_SGDet.jpynb](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/visualization/3.visualize_custom_SGDet.ipynb). The inputs of our visualization code are custom_prediction.json and custom_data_info.json in DETECTED_SGG_DIR. They will be automatically generated if you run the above custom SGDet instruction successfully. Note that there may be too much trivial bounding boxes and relationships, so you can select top-k bbox and predicates for better scene graphs by change parameters box_topk and rel_topk. 

사용자 지정 이미지의 감지된 장면 그래프를 시각화하려면 jupyter 메모를 따르세요. [visualization/3.visualize_custom_SGDet.jpynb](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/visualization /3.visualize_custom_SGDet.ipynb). 시각화 코드의 입력은 DETECTED_SGG_DIR의 custom_prediction.json 및 custom_data_info.json입니다. 위의 사용자 정의 SGDet 명령을 성공적으로 실행하면 자동으로 생성됩니다. 사소한 경계 상자와 관계가 너무 많을 수 있으므로 box_topk 및 rel_topk 매개변수를 변경하여 더 나은 장면 그래프를 위해 top-k bbox 및 술어를 선택할 수 있습니다.


## Other Options that May Improve the SGG

- For some models (not all), turning on or turning off ```MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS``` will affect the performance of predicate prediction, e.g., turning it off will improve VCTree PredCls but not the corresponding SGCls and SGGen. For the reported results of VCTree, we simply turn it on for all three protocols like other models.

- 일부 모델(전부는 아님)의 경우 ```MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS```를 켜거나 끄면 술어 예측 성능에 영향을 미칩니다. 예를 들어, 끄면 VCTree PredCl이 개선되지만 해당 SGCl 및 SGGen은 개선되지 않습니다. 보고된 VCTree 결과의 경우 다른 모델과 같이 세 가지 프로토콜 모두에 대해 간단히 켭니다.

<br>

- For some models (not all), a crazy fusion proposed by [Learning to Count Object](https://arxiv.org/abs/1802.05766) will significantly improves the results, which looks like ```f(x1, x2) = ReLU(x1 + x2) - (x1 - x2)**2```. It can be used to combine the subject and object features in ```roi_heads/relation_head/roi_relation_predictors.py```. For now, most of our model just concatenate them as ```torch.cat((head_rep, tail_rep), dim=-1)```.

- 일부 모델(전부는 아님)의 경우 [Learning to Count Object](https://arxiv.org/abs/1802.05766)에서 제안한 미친 융합이 ```f(x1, x2)와 같이 결과를 크게 향상시킵니다. ) = ReLU(x1 + x2) - (x1 - x2)**2```. ```roi_heads/relation_head/roi_relation_predictors.py```에서 주체와 객체 기능을 결합하는 데 사용할 수 있습니다. 현재 대부분의 모델은 이들을 ```torch.cat((head_rep, tail_rep), dim=-1)```로 연결합니다.

<br>

- Not to mention the hidden dimensions in the models, e.g., ```MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM```. Due to the limited time, we didn't fully explore all the settings in this project, I won't be surprised if you improve our results by simply changing one of our hyper-parameters

- 모델의 hidden dimensions은 말할 것도 없고, 예를 들어 ```MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM```. 제한된 시간 부족으로 이 프로젝트의 모든 셋팅을 완전히 실험하지 못했습니다. 하이퍼파라미터 중 하나를 변경하여 결과를 개선해도 놀라지 않을 것입니다.

## Tips and Tricks for any Unbiased TaskX from Biased Training

The counterfactual inference is not only applicable to SGG. Actually, my collegue [Yulei](https://github.com/yuleiniu) found that counterfactual causal inference also has significant potential in [unbiased VQA](https://arxiv.org/abs/2006.04315). We believe such an counterfactual inference can also be applied to lots of reasoning tasks with significant bias. It basically just runs the model two times (one for original output, another for the intervened output), and the later one gets the biased prior that should be subtracted from the final prediction. But there are three tips you need to bear in mind:

- The most important things is always the causal graph. You need to find the correct causal graph with an identifiable branch that causes the biased predictions. If the causal graph is incorrect, the rest would be meaningless. Note that causal graph is not the summarization of the existing network (but the guidance to build networks), you should modify your network based on causal graph, but not vise versa. 

- For those nodes having multiple input branches in the causal graph, it's crucial to choose the right fusion function. We tested lots of fusion funtions and only found the SUM fusion and GATE fusion consistently working well. The fusion function like element-wise production won't work for TDE analysis in most of the cases, because the causal influence from multiple branches can not be linearly separated anymore, which means, it's no longer an identifiable 'influence'.

- For those final predictions having multiple input branches in the causal graph, it may also need to add auxiliary losses for each branch to stablize the causal influence of each independent branch. Because when these branches have different convergent speeds, those hard branches would easily be learned as unimportant tiny floatings that depend on the fastest/stablest converged branch. Auxiliary losses allow different branches to have independent and equal influences.

counterfactual inference은 SGG에만 적용되는 것은 아닙니다. 사실 제 동료 [Yulei](https://github.com/yuleiniu)는 [unbiased VQA](https://arxiv.org/abs/2006.04315)에서도 counterfactual inference가 상당한 잠재력을 가지고 있음을 발견했습니다. 우리는 그러한 반사실적 추론이 상당한 bias가 있는 많은 추론 작업에도 적용될 수 있다고 믿습니다. 기본적으로 모델을 두 번 실행하고(하나는 원래 출력에 대해, 다른 하나는 개입된 출력에 대해), 나중은 최종 예측에서 빼야 하는 편향된 사전을 얻습니다. 그러나 명심해야 할 세 가지 팁이 있습니다.

- 가장 중요한 것은 항상 인과관계 그래프입니다. 편향된 예측을 유발하는 식별 가능한 분기가 있는 올바른 인과 관계 그래프를 찾아야 합니다. 인과관계 그래프가 올바르지 않으면 나머지는 의미가 없습니다. 인과관계 그래프는 기존 네트워크를 요약한 것이 아니라(네트워크 구축 지침), 인과관계 그래프를 기반으로 네트워크를 수정해야 하지만 그 반대의 경우도 마찬가지입니다.

- causal graph에서 입력 branch가 multiple input branches의 경우 올바른 auxiliary losses(융합 함수)를 선택하는 것이 중요합니다. 우리는 많은 fusion funtion을 테스트했지만 SUM 퓨전과 GATE 퓨전이 일관되게 잘 작동한다는 것을 발견했습니다. element-wise production과 같은 fusion function는 대부분의 경우 TDE 분석에서 작동하지 않습니다. 왜냐하면 multiple branches의 causal influence을 더 이상 선형으로 분리할 수 없기 때문입니다. 즉, 더 이상 식별 가능한 '영향(influence)'이 아닙니다.

- causal graph에 multiple input branches가 있는 최종 예측의 경우 각 독립적 brach의 causal influence를 안정화하기 위해 각 branch에 대한 Auxiliary losses을 추가해야 할 수도 있습니다. 이러한 branches가 서로 다른 수렴 속도를 가질 때, 그 hard branches는 가장 빠르고/가장 안정적인 수렴된 branch에 의존하는 중요하지 않은 작은 float으로 쉽게 학습될 것이기 때문입니다. Auxiliary losses은 다른 branches가 독립적이고 동등한 영향을 가질 수 있도록 합니다.

## Frequently Asked Questions:

1. **Q:** Fail to load the given checkpoints.
**A:** The model to be loaded is based on the last_checkpoint file in the OUTPUT_DIR path. If you fail to load the given pretained checkpoints, it probably because the last_checkpoint file still provides the path in my workstation rather than your own path.

1. **Q:** 주어진 체크포인트를 로드하지 못했습니다.
**A:** 로드할 모델은 OUTPUT_DIR 경로의 last_checkpoint 파일을 기반으로 합니다. 주어진 미리 포함된 체크포인트를 로드하는 데 실패했다면 아마도 last_checkpoint 파일이 여전히 사용자 자신의 경로가 아닌 내 워크스테이션의 경로를 제공하기 때문일 것입니다.

<br>

2. **Q:** AssertionError on "assert len(fns) == 108073"
**A:** If you are working on VG dataset, it is probably caused by the wrong DATASETS (data path) in maskrcnn_benchmark/config/paths_catlog.py. If you are working on your custom datasets, just comment out the assertions.

2. **Q:** "assert len(fns) == 108073"에 대한 AssertionError
**A:** VG 데이터 세트에서 작업하는 경우 maskrcnn_benchmark/config/paths_catlog.py의 잘못된 DATASETS(데이터 경로)로 인해 발생할 수 있습니다. 사용자 정의 데이터 세트에서 작업하는 경우 어설션을 주석 처리하십시오.

<br>

3. **Q:** AssertionError on "l_batch == 1" in model_motifs.py
**A:** The original MOTIFS code only supports evaluation on 1 GPU. Since my reimplemented motifs is based on their code, I keep this assertion to make sure it won't cause any unexpected errors.

3. **Q:** model_motifs.py의 "l_batch == 1"에 대한 AssertionError
**A:** 원래 MOTIFS 코드는 1 GPU에서만 평가를 지원합니다. 다시 구현한 모티프는 해당 코드를 기반으로 하기 때문에 예상치 못한 오류가 발생하지 않도록 이 주장을 유지합니다.

## Citations

If you find this project helps your research, please kindly consider citing our project or papers in your publications.

```
@misc{tang2020sggcode,
title = {A Scene Graph Generation Codebase in PyTorch},
author = {Tang, Kaihua},
year = {2020},
note = {\url{https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch}},
}

@inproceedings{tang2018learning,
  title={Learning to Compose Dynamic Tree Structures for Visual Contexts},
  author={Tang, Kaihua and Zhang, Hanwang and Wu, Baoyuan and Luo, Wenhan and Liu, Wei},
  booktitle= "Conference on Computer Vision and Pattern Recognition",
  year={2019}
}

@inproceedings{tang2020unbiased,
  title={Unbiased Scene Graph Generation from Biased Training},
  author={Tang, Kaihua and Niu, Yulei and Huang, Jianqiang and Shi, Jiaxin and Zhang, Hanwang},
  booktitle= "Conference on Computer Vision and Pattern Recognition",
  year={2020}
}
```
