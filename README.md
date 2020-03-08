## Cycle-consistent Style Transfer (Cycle-GAN)

### Tasks:

The task of this project can be split into two stages:

1. Focus on re-implementing and scaling the pre-trained models from the authors to study and explore various applications of Cycle-GAN.

> For the scope of this project (as we used MacOS) most of the training and implementation has been done in AWS-based Linux AMI, after setup of the dependencies and tools upon GPU.


2. Develop some modifications and a proposal for the architecture. I built a simpler version of the whole model, and the code is accesible in [simple-cg](https://github.com/gvsakashb/cyc-gan/tree/master/simple-cg) directory. 
 
Input: The models take a collection of mages from various datasets, based on our use-case (ususally generated from ImageNet)
Output: An image generated to match the target data, by replicating the style and features as desired.

## Deliverables
* [Code](https://github.com/gvsakashb/cyc-gan/afadf/) 
* Improved model and proposal (simple-cg & ipynb files)
* Final Report
* [AMI image](https://hub.docker.com/repository/docker/mayukuner/text2img), with results / ready to train further.
* Access Key for the image (submitted along with report & results)

* [Dockerfile](Dockerfile)

## Requirements / Prerequisites:
* Linux or macOS
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN

The simpler version harnesses Pytorch with CPU compute power, but can be set to GPU training based on CUDA configuration of local machine.

## Architecture
The baseline model used is CycleGan model, to generate the images in desired style palette from input data. The underlying structure 

Here we use ControlGan as our backbone network to generate high-quality and controllable images from user inputs. The structure of ControlGAN is as follows.
![](https://github.com/mrlibw/ControlGAN/raw/master/archi.jpg)


## Pretrained models

#### Pretrained DAMSM Model
- [DAMSM for bird](https://drive.google.com/file/d/1dbdCgaYr3z80OVvISTbScSy5eOSqJVxv/view?usp=sharing). Download and save it to `DAMSMencoders/`
- [DAMSM for coco](https://drive.google.com/file/d/1k8FsZFQrrye4Ght1IVeuphFMhgFwOxTx/view?usp=sharing). Download and save it to `DAMSMencoders/`

#### Pretrained ControlGAN Model
- [ControlGAN for bird](https://drive.google.com/file/d/1g1Kx5-hUXfJOGlw2YK3oVa5C9IoQpnA_/view?usp=sharing). Download and save it to `models/`
- [ControlGAN for coco](https://drive.google.com/file/d/1Id5AMUFngoZ9Aj-EhMuc590Sv8E3tXjX/view?usp=sharing). Download and save it to `models/`


## Training Phase

To train a ControlGAN, before execute the orders in the instructions below, you should unzip the archive file and then enter the folder `code`.

### [DAMSM](https://github.com/taoxugit/AttnGAN) model includes text encoder and image encoder
- Pre-train DAMSM model for bird dataset:
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0
```
- Pre-train DAMSM model for coco dataset: 
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 0
```
### ControlGAN model 
- Train ControlGAN model for bird dataset:
```
python main.py --cfg cfg/train_bird.yml --gpu 0
```
- Train ControlGAN model for coco dataset: 
```
python main.py --cfg cfg/train_coco.yml --gpu 0
```

`*.yml` files include configuration for training and testing.


## Testing Phase

- Test ControlGAN model for bird dataset:
```
python main.py --cfg cfg/eval_bird.yml --gpu 0
```
- Test ControlGAN model for coco dataset: 
```
python main.py --cfg cfg/eval_coco.yml --gpu 0
```


### Text To Image -- The App


![](imgs/controlgan.gif)

#### Server side

The backend server can be set up on any computer with at least one GPU on it. First, you should use docker to pull down the docker image:

```
docker pull mayukuner/text2img
```

and then run the docker image by:

```
docker run --gpus all -p 5000:5000 mayukuner/text2img
```

Please note, you could also use Dockerfile to build the docker image. But for doing this, the dependencies have to be supported and GPU has to be available aor setup accoridngly.

For the scope of my project, I have trained and optimized the model upon Deep Learning AMI upon EC2 instance, which has delivered promising results.
before doing this, you will have to download the zip file from link provided in [download_code_and_data.txt](download_code_and_data.txt).

Then the server will be successfully set up. The address will be 

```
localhost:5000
```

To get the specific output image from one sentence, you could send a GET request with 3 parameters to the server as:

```
localhost:5000/generate?dataset=<dataset>&sentence=<sentence>&highlight=<word>
```

|           | definition                                                                     | example                                      |
|-----------|--------------------------------------------------------------------------------|----------------------------------------------|
| sentence  | The sentence to generate the image                                             | a herd of cows that are grazing on the grass |
| dataset   | The dataset that the model is trained on                                       | COCO                                         |
| highlight | The highlighted word whose attention  map will be masked on the original image | herd                                         |


The server will respond with a JsonResponse in the form of:

```
{
    “image_url”: <image_url>
}
```

where the variable `<image_url>` indicates the url to the generated image. By further requesting the image file by `<image_url>`, we will get the generated image.

The corresponding output for the example input in the above table is:

![](imgs/example_output.png)

#### Cloud side


Our React-Native App is running on the Expo. 

```
cd text2image_app
yarn install
expo start
```

Download an Expo Client. Open Expo Client on your device. Scan the QR code printed by expo start with Expo Client (Android) or Camera (iOS). You may have to wait a minute while your project bundles and loads for the first time.



### Reference

- Lee, Minhyeok, and Junhee Seok. "Controllable generative adversarial network." IEEE Access 7 (2019): 28158-28169.
- Xu, Tao, et al. "Attngan: Fine-grained text to image generation with attentional generative adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

### Links
- [Cycle-GAN repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

- [Torch-version](https://github.com/junyanz/CycleGAN)

- A PyTorch [colab notebook](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) is also available for exploring and training.
