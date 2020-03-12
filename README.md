## Cycle-consistent Style Transfer (Cycle-GAN)

### Project outline:

1. Intro and pipeline: 

The goal and deliverables of this project relied hevaily on harnessing Cycle-GAN architecture and improving upon the aspects of training and image generation for a particular use case.
> For the scope of this project (as we used MacOS) most of the training and implementation has been done in AWS-based Linux AMI, after setup of the dependencies and tools upon GPU.
 * Scaled and re-implemented the model for horse2zebra, yosemite and style-transfer(Ukiyoe) datasets.
 * Parameter tuning and tabulated results agaisnt varying traina and test conditions. 
  > --model, --netG, --norm, epochs, lr have been explored agaisnt pre-trained and recursive loops for cyclic iterations.
 * Reconstruction and generation of input data has been achieved with improvements.
   a. Detail and crisp/clarity has been observed.
   b. Dehazing and reduction of noise is demonstrated upon introducing the proposed loss function.
   
 
2. Modifications:
A separate directory under [simple-cg](https://github.com/gvsakashb/cyc-gan/tree/master/simple-cg) has modified architecture, (with helper and util functions, pre-trained models narrowed down to 4 .py files). This model has achieved significantly higher detail in images generated and has been used to tabulate and study loss (D, G) for comparative analytics agaisnt change of architecture.

* Pre-trained and reimplementation has some modified code which achieved better with a colo-rbased loss term, thereby tackling issue of hue/tint in style-transfer image data.
* One implementation with pre-initialized weights is also studied to note theat convergence occurs faster.
* Metrics like IoU score, accuracy, quality etc. have achieved comparable results to the BAIR presentation / model.

3. Input: The models take a collection of mages from various datasets.

4. Outputs: An image generated to match the target data, by replicating the style and features of the other data of images.



### Deliverables
* Repository Code - Notebooks & files under 'reimplement'
* Improved model and code ([simple-cg](https://github.com/gvsakashb/cyc-gan/tree/master/simple-cg))
* [AMI image](https://hub.docker.com/repository/docker/mayukuner/text2img), with results / ready to train further.
* Access Key for the EC2 instance / image (*.pem file*)
* [Final report]()

### Requirements / Prerequisites:
* Linux or macOS
* Python 3
* CPU[slower] or NVIDIA GPU + CUDA CuDNN

The simpler version harnesses Pytorch with CPU compute power, but can be set to GPU training based on CUDA configuration of local machine. It is highly suggested to use GPU-supported systems.

### Tasks / Improvement:

The task of this project can be split into two stages:

1. Focus on re-implementing and scaling the pre-trained models from the authors to study and explore various applications of Cycle-GAN.

> For the scope of this project (as we used MacOS) most of the training and implementation has been done in AWS-based Linux AMI, after setup of the dependencies and tools upon GPU.


2. Develop some modifications and a proposal for the architecture. I built a simpler version of the whole model, and the code is accesible in [simple-cg](https://github.com/gvsakashb/cyc-gan/tree/master/simple-cg) directory. 
 
Input: The models take a collection of mages from various datasets, based on our use-case (ususally generated from ImageNet)
Output: An image generated to match the target data, by replicating the style and features as desired.



## Architecture
The baseline model used is CycleGan model, (to generate the images in desired style palette from input and target datasets) is shown below:

<img src="https://github.com/gvsakashb/cyc-gan/blob/master/imgs-readme/G-and-D.png" width="450"><img src="https://github.com/gvsakashb/cyc-gan/blob/master/imgs-readme/network.png" width="450">


#### Observations and results (Re-implementation/pre_trained models):

* 


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

#### Cloud side (AWS)
Most of my work and the implementation was hosted on an AWS AMI. The snapshot is available for copy. There are two ways to set up your environment.

* The Amazon Machine Image (AMI) has been listed in deliverables. Kindly use the same for faster compute & testing.
* Please ensure dependencies of Pytroch and CUDA are working before training / testing.

1. Running from Linux AMI:
	* Pulic AMI-ID: `ami-018b10d93f5a1041e`.
	* Check dependencies before first run.
  ```
  nvcc --version (CUDA check)
  pip install torch torchvision (PyTorch check & install) 
  ```
  * Some of the notebooks have the code (#commented out) for subsequent runs. Please uncomment and run in case of loading and initial run of the files. 

2. Setting your own environemnt
	
* Run the models as given is testing on a smaller subest of images for exploring latency and loss calculations. This can be later expanded to full dataset after trainign of model.
* Tuning and changes to parameters can be done in jupyter as well as .py files based on changes to be made.

> The EC2 instance can be shared for evaluation if neeeded as well. Cloning this repository on a machine with GPU support is an ideal way to run locally, but AMI can be used to study my results.


### Reference

- Lee, Minhyeok, and Junhee Seok. "Controllable generative adversarial network." IEEE Access 7 (2019): 28158-28169.
- Xu, Tao, et al. "Attngan: Fine-grained text to image generation with attentional generative adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

### Links
- [Cycle-GAN repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Torch-version](https://github.com/junyanz/CycleGAN)
- GCP [colab notebook](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb).
- A version to study the baseline architecture and plots of loss functions is [available](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/cycle-gan).
- bair.berkeley.edu/blog/2019/12/13/humans-cyclegan (BAIR blog with recent updates)
