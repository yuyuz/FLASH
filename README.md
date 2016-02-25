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

To avoid a variety of potential problems in environment settings, we highly recommend to use Miniconda (Python 2.7).

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

Now we install dependencies within conda environment:
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

All the benchmark datasets are publicly available [here](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets). These datasets were first introduced by [Auto-WEKA](http://www.cs.ubc.ca/labs/beta/Projects/autoweka) and have been widely used to evaluate Bayesian optimization methods.

Due to the file size limit, we are not able to provide all those datasets in our Github repository. In fact, only the ``madelon`` dataset is provided as an example. To deploy a new benchmark dataset, download the zip file from [here](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets) and then uncompress it. You will get a dataset folder including two files ``train.arff`` and ``test.arff``. Move this folder into the [data directory](https://github.com/yuyuz/FLASH/tree/master/data), just like the dataset folder ``madelon`` we already put there.

## How to Run?

For benchmark datasets, we build a general data analytic pipeline based on [scikit-learn](http://scikit-learn.org), following the pipeline design of [auto-sklearn](https://github.com/automl/auto-sklearn). We have 4 computational steps with 33 algorithms in this pipeline. Details are discussed in the [paper](http://arxiv.org/abs/1602.06468).

To run this pipeline on a specific dataset, first you need to correctly set the configuration file (``/path/to/FLASH/benchmarks/sklearn/config.cfg``):

* In the ``HPOLIB`` section, change the ``function`` path according to your local setting.
* In the ``HPOLIB`` section, change the ``data_path`` according to your local setting.
* In the ``HPOLIB`` section, change the ``dataset`` name to whichever dataset you have deployed as the input of pipeline.

Now you can tune the pipeline using different Bayesian optimization methods. For each method, we provide a Python script to run the tuning process.

For our method, it currently has two versions (with different optimizers in the last phase): **FLASH** and **FLASH<sup>*</sup>**.  
To run **FLASH**:
```bash
cd /path/to/FLASH/benchmarks/sklearn
python run_flash.py
```

To run **FLASH<sup>*</sup>**:
```bash
cd /path/to/FLASH/benchmarks/sklearn
python run_flash_star.py
```

For other methods ([SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC), [TPE](http://jaberg.github.io/hyperopt), Random Search), we use the implementations in [HPOlib](http://www.automl.org/hpolib.html) and also provide Python scripts.  
To run SMAC:
```bash
cd /path/to/FLASH/benchmarks/sklearn
python run_smac.py
```

To run TPE:
```bash
cd /path/to/FLASH/benchmarks/sklearn
python run_tpe.py
```

To run Random Search:
```bash
cd /path/to/FLASH/benchmarks/sklearn
python run_random.py
```

## Advanced Configurations

In the configuration file (``/path/to/FLASH/benchmarks/sklearn/config.cfg``), you can set quite a few advanced configurations.

The configuration items in the ``HPOLIB`` section are effective for all the optimization methods above:

* Set ``use_caching`` as ``1`` to enable pipeline caching, ``0`` to disable caching.
* ``cv_folds`` specifies the number of cross validation folds during the optimization.
* For other items such as ``number_of_jobs`` and ``result_on_terminate``, refer to the HPOlib [manual](http://www.automl.org/manual.html).

The configuration items in the ``LR`` section are only effective for FLASH and FLASH<sup>*</sup>:

* Set ``use_optimal_design`` as ``1`` to enable optimal design for initialization, ``0`` to use random initialization.
* ``init_budget`` specifies the number of iterations for Phase 1 (initialization).
* ``ei_budget`` specifies the number of iterations for Phase 2 (pruning).
* ``bopt_budget`` specifies the number of iterations for Phase 3 (fine-tuning).
* ``ei_xi`` is the trade-off parameter ξ in EI and EIPS functions, which balances the exploitation and
exploration.
* ``top_k_pipelines`` specifies the number of best pipeline paths to select at the end of Phase 2 (pruning).
