### Simpler version

* While deploying and training the official models against various datasets, a simpler archcitecture was developed.

* Idea:

<img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/results/idea.png" width=300>

* Instead of implementing multiple things at once, we can make an easier implementation as shown in this sub-directory (simple-cg).


Dependencies: Same as the original implementation, but code relies more on Python and Pytorch, working at reduced GPU compute power. 

#### Implementation and run:

* download data: 
> $ sh ./download_dataset.sh horse2zebra
* Training: 
> $ python main.py --training True
* Testing:
> $ python main.py --testing True

* Losses tabulated here enabled in improving and tuning of original model to optimize the parameters to achieve best reuslts.

<img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/losses.png" height="300">

#### Experimentation: 

Along with changes done while training upon the original repo code, changes between batch and instance normalization, modifying the content_loss (pixel / vgg) parameters has been studied. 
* Tweaking these arguments and the obtained results are documented in the final report.


#### Some results:

* For h2z data:

( Real - Generated - Reconstructed)

<p float="left">
  <img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/results/horse_real.png" width="200" />
  <img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/results/zebra_generated.png" width="200" />
  <img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/results/horse_reconstructed.png" width="200" />
</p>

Batch (--norm) / vgg implementation:

<img src="https://github.com/gvsakashb/cyc-gan/blob/master/simple-cg/results/batch.png" width="400">

> Links:
* Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
* Original Pytorch repo: [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* This has been done extending upon the University of Toronto's assignment on cycle-gan and referring to some implementations on Tensorlow and torch in the github community.
