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
