In this assignment you will practice writing backpropagation code, and training
Neural Networks and Convolutional Neural Networks. The goals of this assignment
are as follows:

- understand **Neural Networks** and how they are arranged in layered
  architectures
- understand and be able to implement (vectorized) **backpropagation**
- implement various **update rules** used to optimize Neural Networks
- implement **batch normalization** for training deep networks
- implement **dropout** to regularize networks
- effectively **cross-validate** and find the best hyperparameters for Neural
  Network architecture
- understand the architecture of **Convolutional Neural Networks**
- gain an understanding of how a modern deep learning library (PyTorch) works
  and gain practical experience using it to train models.

## Setup
Make sure your machine is set up with the assignment dependencies.

### [Option 1] Install Anaconda and Required Packages
The preferred approach for installing all the assignment dependencies is to use
[Anaconda](https://www.anaconda.com/products/individual), which is a Python distribution
that includes many of the most popular Python packages for science, math,
engineering and data analysis. Once you install Anaconda you can run the following
command inside the homework directory to install the required packages for this homework:

```bash
conda env create -f cs182_hw1.yaml
```

Once you have all the packages installed, run the following command **every time**
to activate the environment when you work on the homework.
```bash
conda activate cs182_hw1
```

### [Option 2] Working through Docker
We've designed a Docker container that has all requisite packages installed + should work for Windows and Mac.

To use this approach, install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your OS. After launching, either create an account or log in with your Github account. Then, go on the left sidebar and click on Images > Hub > Search > `verityw/datac182`. After downloading the `hw1` image (which may take some time, on account of the file being ~10 GB), you should be able to start up a container by opening a terminal in Docker Desktop (bottom right corner) and running:
```bash
docker run -it -p 8888:8888 -v /PATH/TO/HW/REPO:/contained verityw/datac182:hw1 bash
```
**Please make sure to replace `/PATH/TO/HW/REPO` in the above command with the actual path to the homework repo on your local machine!!!** If you get some sort of permission error, you'll need to [enable file-sharing for mounting directories in Docker Desktop](https://forums.docker.com/t/automatically-permit-sharing-for-mounts/141124/3).

This will also connect port 8888 on your host machine to 8888 in the container (you can change either of these if needed). Additionally, it mounts your homework repo to a directory inside the container called `/contained` (any files in your homework repo will be accessible both inside and outside the container). **These are the only persistent files: make sure anything you want to save goes in that mounted directory, since all other files will be deleted when you stop the Docker container!** 

This allows you to enter the Docker container through an interactive terminal, which you can use as a normal terminal. You can use this to run e.g., the data download script (if you get a permission denied error, run `chmod u+r+x get_datasets.sh` before running the script again). Likewise, you should be able to compile the Cython stuff (if it says that it's trying to call gcc but can't find it, just run apt-get update and apt-get gcc . If it says something about assigning double to int, check the HW1 announcement thread for a fix / re-pull the repo).

Finally, after entering your homework repo in the container run:
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```
This will start the Jupyter server, which you'll be able to use on your local machine by going to `localhost:8888` in your browser. You'll be able to run the notebooks for the homework this way!

### [Option 3] [WIP] Working on a Virtual Machine
This assignment is provided pre-setup with a VirtualBox image. Installation Instructions:
1. Follow [the instructions here](https://www.virtualbox.org/manual/ch02.html) to install VirtualBox if it is not already installed.
2. [Download the VirtualBox image here](https://drive.google.com/file/d/1J1oRKQtIa5gMUxpzOrdLL6taQtGA4a3k/view?usp=sharing)
3. Load the VirtualBox image using [the instructions here](https://docs.oracle.com/cd/E26217_01/E26796/html/qs-import-vm.html)
4. Start the VM. The username and password are both cs182. Required packages are pre-installed and the cs182_hw1 environment activated by default.
5. Download the assignment code onto the VM yourself.

### Download Data
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from this repo:

```bash
cd deeplearning/datasets
./get_datasets.sh
```

If running on Windows (where you can't run bash scripts), please run the above inside the Docker interactive terminal.

### Compile the Cython extension
Convolutional Neural Networks require a very
efficient implementation. We have implemented of the functionality using
[Cython](http://cython.org/); you will need to compile the Cython extension
before you can run the code. From the `deeplearning` directory, run the following
command:

```bash
python setup.py build_ext --inplace
```

### Start Jupyter Notebook
After you download data, you should start the IPython notebook server
from the homework 1 directory with the following command:

```bash
jupyter notebook
```
This opens a brower tab for you to work.

If you are unfamiliar with IPython, you should
read our [IPython tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).

## Submitting your work:
Once you are done working run the `collect_submission.sh` script;
this will produce a file called `assignment1.zip`.
Upload this file to Gradescope.
The Gradescope will run an autograder on the files you submit. It is very unlikely but still possible that your implementation might fail to pass some test cases due to randomness.
If you think your code is correct, you can simply rerun the autograder to check check whether it is really due to randomness.

### Q1: Fully-connected Neural Network (35 points)
The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

### Q2: Batch Normalization (25 points)
In the IPython notebook `BatchNormalization.ipynb` you will implement batch
normalization, and use it to train deep fully-connected networks.

### Q3: Dropout (10 points)
The IPython notebook `Dropout.ipynb` will help you implement Dropout and explore
its effects on model generalization.

### Q4: ConvNet (20 points)
In the IPython Notebook `ConvolutionalNetworks.ipynb` you will implement several
new layers that are commonly used in convolutional networks as well as implement
a small convolutional network.

### Q5: Train a model on CIFAR10 using Pytorch! (10 points)
Now that you've implemented and gained an understanding for many key components
of a basic deep learning library, it is time to move on to a modern deep learning
library: Pytorch. Here, we will walk you through the key concepts of PyTorch, and
you will use it to experiment and train a model on CIFAR10.

If you would like to access GPUs for faster training, we recommend
you use Google Colab (https://colab.research.google.com/). If you use Colab for this notebook, make sure to manually download the completed
notebook and place it in the assignment directory before submitting. Also remember
to download required output file and place it into submission_logs/ directory.

### FAQ

**I'm getting some weird issue about Cython assigning float to int** 

This is because Cython (which compiles Python to C) thinks that the code you provided is in Python 3 instead of Python 2. This is a problem because Python 2 converts int / int to int, while Python 3 makes it a float. By default, it should set to Python 2 (even if the actual Python version you're using is 3) -- I'm not sure why it isn't doing that for you. You can force it to use Python 2 by going into deeplearning/setup.py and looking for the part:

```python
setup(
    ext_modules=cythonize(extensions),
)
```
This is what compiles im2col_cython.pyx into C. You can add `compiler_directives={'language_level' : "2"}` inside the `cythonize()` to force it to use Python 2, see [here](https://stackoverflow.com/questions/34603628/how-to-specify-python-3-source-in-cythons-setup-py).

**Cython gives an about a file called longintrepr.h not being found** 

Please lower your Python version to < 3.11 (we suggest 3.8 or 3.9). This should not be an issue for anyone using our Conda yaml file (or the Docker), since it sets the Python version to one of those versions.

**I get an error "AMD-V is disabled in the BIOS" or "Intel-VT is disabled in the BIOS" or similar**

Solution: See [this link](https://docs.fedoraproject.org/en-US/Fedora/13/html/Virtualization_Guide/sect-Virtualization-Troubleshooting-Enabling_Intel_VT_and_AMD_V_virtualization_hardware_extensions_in_BIOS.html)


**The virtual machine won't boot**

Solutions:

- Try increasing the number of allocated CPUs: Under Settings→System→Processor
- Try [increasing the amount of allocated memory:](https://superuser.com/questions/926339/how-to-change-the-ram-allocated-to-an-os-in-virtualbox)

