# Neural radiance fields-based multi-view endoscopic scene reconstruction for surgical simulation
We propose an Endoscope-NeRF network for implicit radiance fields reconstruction of endoscopic scene under non-fixed light source, and synthesize novel views.

This is the official repo for the implementation:

## Usage
Clone this repository:
```
git clone --recurse-submodules https://github.com/qinzhibao123/Endoscope-NeRF-main
cd Endoscope-NeRF-main/

```

Dependencies:
The code is implemented with Python3.7, PyTorch == 1.5 and CUDA == 10.2. 
```
conda env create -f environment.yml
conda activate Endoscope-NeRF
```

## Datasets

```
├──data/
    ├──pretrain_data_denoising/
        ├──scene1/
            ├──images/
            ├──poses_bounds.npy
        ...
    ├──Endoscope_data/
        ├──scene1/
            ├──images/
            ├──poses_bounds.npy
        ...
```
Please first `cd data/`, and then download datasets ([IRON](https://github.com/Kai-46/IRON) and Ours) into `data/`. Here the poses_bounds.npy follows the data format in [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch),

## Training
We train the model with a single GPU:
```
python train.py --config configs/pretrain.txt
```

## Finetuning
Fine-tuning the pretrained model on a new endoscopic scene, run:
```
python train.py --config configs/finetune_llff.txt
```

## Evaluation
We evaluate the models fine-tuned on each scene to obtain evaluation metrics (PSNR, SSIM, and LPIPS) for synthesized images, run:
```
cd eval/
python eval.py --config ../configs/eval_llff.txt
``` 

## Rendering videos of smooth camera paths
Rendering the video under a smooth camera path, run:
```
cd eval/
python render_llff_video.py --config ../configs/eval_llff.txt
```


