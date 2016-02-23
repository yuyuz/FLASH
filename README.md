# FLASH

## What is FLASH?

FLASH is a package to perform Bayesian optimization on tuning data analytic pipelines. Specifically, it is a two-layer Bayesian optimization framework, which first uses a parametric model to select promising algorithms, then computes a nonparametric model to fine-tune hyperparameters of the promising algorithms.

Details of FLASH are described in the paper:

**FLASH: Fast Bayesian Optimization for Data Analytic Pipelines** [[arXiv](http://arxiv.org/abs/1602.06468)]  
Yuyu Zhang, Mohammad Taha Bahadori, Hang Su, Jimeng Sun

FLASH is licensed under the GPL license, which can be found in the package.

## Installation

FLASH is developed on top of [HPOlib](http://www.automl.org/hpolib.html), a general platform for hyperparameter optimization. Since HPOlib was developed on Ubuntu and currently only supports Linux distributions, FLASH also only works on Linux (we developed and tested our package on Ubuntu).

**1. Clone repository**
```bash
git clone https://github.com/yuyuz/FLASH.git
```

**2. Install Miniconda**

To avoid a variety of potential problems in environment settings, we highly recommend to use Miniconda.

If you are using 64-bit Linux system (recommended):
```bash
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh
```

If you are using 32-bit Linux system:
```bash
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86.sh
bash Miniconda-latest-Linux-x86.sh
```

Answer ``yes`` when the installer asks to prepend the Miniconda2 install location to PATH in your ``.bashrc``. 

After installation completed, restart terminal or execute ``source ~/.bashrc`` to make sure that conda has taken charge of your Python environment.

**3. Install dependencies**

Now we install dependencies within conda environment.
```bash
easy_install -U distribute
conda install -y openblas numpy scipy matplotlib scikit-learn==0.16.1
pip install hyperopt liac-arff
```

**4. Install package**

Install HPOlib and some requirements (``pymongo``, ``protobuf``, ``networkx``). During the installation, please keep your system **connected to the Internet**, such that ``setup.py`` can download optimizer code packages.
```bash
cd /path/to/FLASH
python setup.py install
```

## Benchmark Datasets

All the benchmark datasets are publicly available [here](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets/). These datasets were first introduced by [Auto-WEKA](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/) and have been widely used to evaluate Bayesian optimization methods.

Due to the file size limit, we are not able to provide all those datasets in our Github repository. To deploy a benchmark dataset, just download the zip file from [here](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets/) and uncompress it. You will get a dataset folder including two files ``train.arff`` and ``test.arff``. Move this folder into our ``data`` directory. Now you can run the pipeline on this dataset.

## How to Run?

For benchmark datasets, we build a general data anlalytic pipeline based on [scikit-learn](http://scikit-learn.org/). Details about this pipeline are described in the [paper](http://arxiv.org/abs/1602.06468).
