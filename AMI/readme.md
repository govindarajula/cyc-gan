#### The Amazon Virtual (Machine) image is also available upon an EC2 instance, and the key is in AMI folder of the repo. 

> Please mention and it can be shared, but all the files utilized / built have been pushed to GitHub repo. 
(gvsakashb@u.northwestern.edu)

*	Using the AMI file and working on Amazon console is ideal to evaluate in a CPU-based system. If in a GPU-supported computer, local training can be done.
*	Connecting to EC2 (shell/cmd): 
1. `chmod 400 adl-gan.pem`
2.	`ssh -L localhost:8888:localhost:8888 -i adl-gan.pem ubuntu@<Public DNS>`
3. 	Run the python files for preprocessing, if needed and check dependencies before using Jupyter.

AMI typically needs AWS credits or working with the instance generated from AMI file, please ensure all the requiremnts are met. This project cna be trained and run locally, with the systems having GPU support.
