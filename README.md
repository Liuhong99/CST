# CST
Code release for "Cycle Self-Training for Domain Adaptation" (NeurIPS 2021)

## Prerequisites
- torch>=1.7.0
- torchvision
- qpsolvers
- numpy
- prettytable
- tqdm
- scikit-learn
- webcolors
- matplotlib


## Training

VisDA-2017
```
CUDA_VISIBLE_DEVICES=0 python run_cst.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 \
--epochs 30 --early 12 --lr 0.002 --per-class-eval --temperature 3.0 --center-crop --log logs/cst/VisDA2017 \
--trade-off 0.08 trade-off1 2.0 --trade-off3 0.5 --threshold 0.97 -b 28 
```

Office Home
```
CUDA_VISIBLE_DEVICES=0 python run_cst.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 \
--epochs 30 --early 30 --temperature 2.5 --bottleneck-dim 2048 --log logs/cst/OfficeHome_Pr2Rw \
--trade-off1 2.0 --trade-off3 0.5 --threshold 0.97 --trade-off 0.015
```


## Acknowledgement
This code is implemented based on the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library/), and it is our pleasure to acknowledge their contributions.

The SAM code is adapted from [https://github.com/davda54/sam](https://github.com/davda54/sam).



## Citation
If you use this code for your research, please consider citing:
```
@article{liu2021cycle,
  title={Cycle Self-Training for Domain Adaptation},
  author={Liu, Hong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2103.03571},
  year={2021}
}
```

## Contact
If you have any problem about our code, feel free to contact
- h-l17@tsinghua.org.cn