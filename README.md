<img src='imgs/suit.gif' align="right" width=384>

<br><br><br>

# Unpaired Pose-Guided Human Image Generation


## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone git@github.com:cx921003/UPG-GAN.git
cd UPG-GAN
```
### Data Preparation
- Download a dataset from our [Google Drive](https://goo.gl/KRQ9tM).
- Unzip the dataset under ``./datasets/`` folder.

### Pre-trained Models
- Download a pre-trained model from our [Google Drive](https://goo.gl/YwcWvv).
- Unzip the model under ``./checkpoints/`` folder.

### Testing:
- Configure the following arguments in ``./testing.sh``:
    - ``dataroot``: the path to the dataset
    - ``name``: the name of the model, make sure the model exists under ``./checkpoint/``
    - ``how_many``: number of input images to test
    - ``n_samples``: number of samples per input image
    
    
### Training
- Configure the following arguments in ``./training.sh``:
    - ``dataroot``: the path to the dataset
    - ``name``: the name of the model
- Train a model:``./training.sh``
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/suit_and_dress/web/index.html`



The test results will be saved to a html file here: `./results/suit_and_dress/latest_test/index.html`.




## Citation
If you use this code for your research, please cite our papers.
```
to be added
```

## Acknowledgments
Code is heavily based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git) written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung89).
