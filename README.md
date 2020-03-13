## Cycle-consistent Style Transfer (Cycle-GAN)

[Original project](https://junyanz.github.io/CycleGAN/) | [Paper](https://arxiv.org/pdf/1703.10593.pdf) | [original repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### Project outline

1. Intro: 

The goal and deliverables of this project relied heavily on harnessing Cycle-GAN architecture and improving the architecture.
> For the scope of this project (as we used MacOS) most of the training and implementation has been done in AWS-based Linux AMI, after setup of the dependencies and tools upon GPU.
 * Scaling and re-implementation has been done for horse2zebra, yosemite and style-transfer(Ukiyoe) datasets using the model.
 * Parameter tuning against varying train and test conditions is studied and results are tabulated. 
  > --model, --netG, --norm, epochs, lr-plocy, etc. have been explored against pre-trained and recursive loops for cyclic iterations.
 * Reconstruction and generation of input data has been achieved with improvements (clarity/resolution, dehazing and noise reduction).
   
2. Modifications:
A separate directory under [simple-cg](https://github.com/gvsakashb/cyc-gan/tree/master/simple-cg) has modified architecture, (with helper and util functions, pre-trained models narrowed down to 4 .py files). This model has achieved significantly higher detail in images generated and has been used to tabulate and study loss (D, G) for comparative analytics against changes of baseline architecture.

* Pre-trained and reimplementation has some modified code which achieved better with a colo-rbased loss term, thereby tackling issue of hue/tint in style-transfer image data.
* One implementation with pre-initialized weights is also studied to note theat convergence occurs faster.
* Metrics like IoU score, accuracy, quality etc. have achieved comparable results to the BAIR presentation / model.

3. Input: The models take a collection of mages from various datasets.

4. Outputs: An image generated to match the target data, by replicating the style and features of the other data of images.


### Architecture
The baseline model used is CycleGan model, (to generate the images in desired style palette from input and target datasets) is shown below:

<img src="https://github.com/gvsakashb/cyc-gan/blob/master/imgs-readme/G-and-D.png" width="400"><img src="https://github.com/gvsakashb/cyc-gan/blob/master/imgs-readme/network.png" width="400">


#### Deliverables
* Repository Code - Notebooks & files under 'reimplement'
* Improved [simple-cg](https://github.com/gvsakashb/cyc-gan/tree/master/simple-cg) model 
* AMI image (with results and ready to train further)
* [Final report](file.pdf)
* Access Key for the EC2 instance / image (*.pem file*)

#### Requirements / Prerequisites:
* Linux or macOS
* Python 3
* PyTorch
* NVIDIA GPU + CUDA CuDNN

> The simpler version harnesses Pytorch with CPU, but can be set to GPU training based on CUDA. 

> It is highly suggested to use GPU for the original models. In case of no GPU support, AMI provided can be used.

---

<img src='/imgs-readme/h2z.gif' align="right" width=300>

#### Cloning and use:
Clone this repo:
```
git clone https://github.com/gvsakashb/cyc-gan
cd cyc-gan
```
Install the dependencies:
* cd into reimplement to check for dependencies `cd reimplement`
* For pip users, please type the command `pip install -r requirements.txt`.
* Install [PyTorch](pytorch.org) and ensure CUDA is setup on your device with necessary support `nvcc --version` (verify).
* Run the jupyter notebooks from reimplement or the other simpler directory to study and evaluate the models.
* Tuning and chanegs can be done in jupyter(examples commented in code) or in the supporting .py file in the directory of use.

#### Training Phase

To train a CycleGAN, before execute the orders in the instructions below, the approach varies if you cloned or are running on AMI.



##### Pre-trained models can be loaded.

Preprocessing involves harnessing the scripts provided from original directory for training and setup of our model.
```
!bash ./scripts/download_cyclegan_model.sh monet2photo
```
The datset can be specified at the end as shown. This can be done pre-emptively or in jupyter before start of training.
* Reimplementation has mostly capitalized on this approach, with modfications and tuning of parameters is done for train.py and the scripts files.
* Run of pre-trained models has a lot of noise and this is improved by the changes done and training over a large amount of images with more epochs.

<img src='https://github.com/gvsakashb/cyc-gan/blob/master/imgs-readme/style-initial.png' height=250>

##### Train:

<img src="imgs-readme/logic.png" width="500"> <img src="imgs-readme/reconstruct.png" width="350">
* 
```
!python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan
```

* For the newer version (simple-cg), the readme inside the folder details more about how to train the model, and tweaks can be done in train.py file.
* This model avoids the pre-trained and options/util and other files and provides more incentive to change and tune for the user, hence simplifying approach as we generate better results at improved latency.

* Upon exploring the under-the-hood architecture, [Adam](https://pytorch.org/docs/stable/optim.html#algorithms) optimizers and loss functions are narrowed down from code and these are modified and studied.
* A CycleGAN repeats its training process, alternating between training the discriminators and the generators, for a specified number of training iterations.

#### Testing Phase

* For the cloned repo, the test can be done with: 
```
!python test.py --dataroot datasets/summer2winter_yosemite/testA --name summer2winter_yosemite_pretrained --model test --no_dropout
```
* Parameter tuning and dropout, lr-policy all have been modified in various re-runs to study changes for each.
* Usage of cycle-gan in model recursively has been studied to see anyy optimization adn improvement in image generation. These are shown in our report.
* The dataroot and name can be changed for desired dataset and tested accordingly. 
* The best images are saved for reference in results directory. A separate folder is also provided to illsutrate improvements observed.

* For the newer version (simple-cg), the readme inside the folder details more about how to run and test the model.

#### Cloud side - AMI (AWS)
* The Amazon Machine Image (AMI) has been listed in deliverables. Kindly use the same for faster compute & testing.
* Please ensure dependencies of Pytroch and CUDA are working before training / testing.

Most of my work and the implementation was hosted on an AWS Deep Learning AMI. The snapshot is available for copy. There are two ways to set up your environment:

1. Running from Linux AMI
	* Pulic AMI-ID: `ami-018b10d93f5a1041e`.
	* Check dependencies before first run.
  	```
	nvcc --version
	pip install torch torchvision 
	```
	* Some of the notebooks have the code (#commented out) for subsequent runs. Please uncomment and run in case of loading and initial run of the files. 
  	> This can be checked based on errors showing loading of pre-trained models.

2. Setting your own environemnt
* After opening AMI, the environment can be setup with dependencies & tested with a smaller subest of images for exploring results and loss calculations. 
* This can be later expanded to the full dataset. Tuning and changes to parameters can be done in jupyter as well as .py files.

> The EC2 instance can be shared for evaluation if neeeded. Cloning this repository on a machine with GPU support is an ideal way to run locally, but the AMI can be used to study my results.


#### Observations and Results

* Please refer the [report](report/adl-report.pdf) for a detailed explanation about the changes in code and results.
* Specific to simple-cg, the training results are better for instance normalization, with better clarity in images for subsequent model runs.
* The [folder](/best-results) has some of the best generated images from various iterations and re-runs/designed models. Kindly check the same to see some interesting results and improvements in contrast to a typical/baseline model.

### Links and References
- [Torch-version](https://github.com/junyanz/CycleGAN)
- [BAIR blog](bair.berkeley.edu/blog/2019/12/13/humans-cyclegan) with recent updates.
- Some other papers studied are linked in the [explore.md](https://github.com/gvsakashb/cyc-gan/blob/master/explore.md) file.

### Citation:
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
