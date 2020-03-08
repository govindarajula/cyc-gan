### Simpler version in contrast to official-repo

* While deploying and training the official models against various datasets, a simpler arhcitecture of the model was explored.
* Instead of implemeneting multiple things at once, we can make an easier implementation as shown in this sub-directory(simple-cg).

Dependencies: Same as the original implementation, but code is based on Python and Pytorch.(0.4.1)
#### Implementation and run:

* download data: 
> $ sh ./download_dataset.sh horse2zebra
* Training and testing: 

> $ python main.py --training True

> $ python main.py --testing True

#### Experimentation: 

Along with changes done while training upon the original repo code, changes between batch and instance normalization, modifying the content_loss (pixel / vgg) parameters has been studied. Tweaking these arguments and the obtained results are docuemnted in the report and the final presentation for reference.

#### Some results:

* For h2z data:

( Real - Generated - Reconstructed)

<p float="left">
  <img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/images/horse_real.png" width="200" />
  <img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/images/zebra_generated.png" width="200" />
  <img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/images/horse_reconstructed.png" width="200" />
</p>

> Links:
* Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
* Original Pytorch repo: [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* This has been done extending upon the University of Toronto's assignment on cycle-gan and referring to some implementations on Tensorlow and torch in the github community.
