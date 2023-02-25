# PPCL

PyTorch implementataion for [Proactive Privacy-preserving Learning for Cross-modal Retrieval](https://dl.acm.org/doi/full/10.1145/3545799)



## Environment

pip install -r requriements.txt

```
h5py==3.8.0
numpy==1.23.5
opencv_python==4.7.0.68
Pillow==9.4.0
scipy==1.10.1
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.1
```

## Train

Download the [MIRFLICKR25K](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew)(password:8dub) and put it into `./datasets`.then run

```python
python train.py
```

The training process costs about 30min on 1*GeForce RTX 3090.

## Test

JDSH is selected as the targeted fooling retrieval system to test the validity of PPCL.

Download the [checkpoint file of PPCL](https://drive.google.com/file/d/1CoOZCFlC5hfrWOhPyFX9cd9tk5oRGgSz/view?usp=share_link) and [well-trained JDSH model](https://drive.google.com/file/d/1LDKsBhSJgtJs0SXBcocuyUyIbNAVFDsY/view?usp=share_link) based on [JDSH repo](https://github.com/KaiserLew/JDSH) and put them into `./models`,then run

```
python eval.py
```



## Results on JDSH

| Categories                            | MAP       |
| ------------------------------------- | --------- |
| **MAP of Image to Text**              | **0.775** |
| **MAP of Image to Image**             | **0.824** |
| **MAP of Text to Image**              | **0.847** |
| **MAP of Text to Text**               | **0.833** |
| **MAP of Image to Noise Image**       | **0.703** |
| **MAP of Noise Image to Image**       | **0.684** |
| **MAP of Noise Image to Noise Image** | **0.683** |
| **MAP of Text to Noise Image**        | **0.724** |
| **MAP of Noise Image to Text**        | **0.663** |



## Note

The retrieval model takes Alexnet as the backbone instead of VGG16,which is slightly different from the paper.

