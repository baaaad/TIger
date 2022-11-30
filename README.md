# TIger
This repository contains the reference code for the ECE model TIger proposed in the paper [Explicit Image Caption Editing](https://arxiv.org/abs/2207.09625) accpeted to ECCV 2022. Refer to our full paper for detailed intructions and analysis. The dataset and more detailed task information are available in this [ECE repository](https://github.com/baaaad/ECE).

Please cite with the following BibTeX:

```
@inproceedings{wang2022explicit,
  title={Explicit Image Caption Editing},
  author={Wang, Zhen and Chen, Long and Ma, Wenbo and Han, Guangxing and Niu, Yulei and Shao, Jian and Xiao, Jun},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXVI},
  pages={113--129},
  year={2022}
}
```
![model](images/TIger.png)


## Environment setup
Clone the repository and create the `tiger` conda environment using the `conda.yml` file:
```
conda env create -f conda.yml
conda activate tiger
```

## Data preparation
### COCO_EE and Flickr30K_EE
The processed datasets have been placed in the [dataset](https://github.com/baaaad/TIger/tree/main/datasets) folder, they can also be directly download from [here](https://drive.google.com/drive/folders/1nzIsGT4SC81aMcC48tCMWcqL77sgYrvT?usp=sharing), including the COCO-EE and Flickr30K-EE in `train`, `dev` and `test` splits.

### Visual Features
For visual token features, we used the bottom-up features (36 regions for each image) which are extracted by a pre-trained Faster R-CNN. 

**COCO-EE**
Download the pre-computed features file [coco_36.tsv](http://ailb-web.ing.unimore.it/releases/show-control-and-tell/coco_detections.hdf5) (~45.2 GB) and place it under the `datasets/bottom_up/COCO_36` folder, run the 'process_bottom_up_feature.py' to process the feature.
```
python process_bottom_up_feature.py
```

**Flickr30K-EE**
Download the pre-computed features file [flickr30k_36.tsv](http://ailb-web.ing.unimore.it/releases/show-control-and-tell/coco_detections.hdf5) (~11.6 GB) and place it under the `datasets/bottom_up/Flickr30K_36` folder, run the 'process_bottom_up_feature.py' to process the feature.
```
python process_bottom_up_feature.py
```
