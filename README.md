# Okapi: Generalising Better By Making Statistical Matches Match

Official code for the NeurIPS 2022 paper _Okapi: Generalising Better By Making
Statistical Matches Match_

> We propose Okapi, a simple, efficient, and general method for robust
semi-supervised learning based on online statistical matching. Our method uses
a nearest-neighbours-based matching procedure to generate cross-domain views
for a consistency loss, while eliminating statistical outliers. In order to
perform the online matching in a runtime- and memory-efficient way, we
draw upon the self-supervised literature and combine a memory bank with
a slow-moving momentum encoder. The consistency loss is applied within
the feature space, rather than on the predictive distribution, making
the method agnostic to both the modality and the task in question. We
experiment on the WILDS 2.0 datasets Sagawa et al., which significantly
expands the range of modalities, applications, and shifts available for
studying and benchmarking real-world unsupervised adaptation. Contrary
to Sagawa et al., we show that it is in fact possible to leverage
additional unlabelled data to improve upon empirical risk minimisation
(ERM) results with the right method. Our method outperforms the
baseline methods in terms of out-of-distribution (OOD) generalisation
on the iWildCam (a multi-class classification task) and PovertyMap (a
regression task) image datasets as well as the CivilComments (a binary
classification task) text dataset. Furthermore, from a qualitative
perspective, we show the matches obtained from the learned encoder are
strongly semantically related.

## Requirements
- python >=3.9
- [poetry](https://python-poetry.org/)
- CUDA >=11.3 (if installing with ``install.sh``)

## Installation
We use [poetry](https://python-poetry.org/) for dependency management,
installation of which is a prerequisite for installation of the python
dependencies. With poetry installed, the dependencies can then be installed by
running ``install.sh``, contingent on CUDA >=11.3 being installed if installing
to a CUDA-equipped machine. This constraint can be bypassed by manually
excuting the commands:
- ``poetry install``
- install the appropriate version of Pytorch and ``torch-scatter`` (required
  for evaluation with [WILDS](https://github.com/p-lambda/wilds)) for the
  version of CUDA installed on your machine.

## Running the code
We use [hydra](https://github.com/facebookresearch/hydra) for managing the
configuration of our experiments. Experiment configurations are grouped by
dataset in ``external_confs/experiments`` and can be imported via the
commandline with the command ``python main.py +experiment={dataset}/{method}``;
one can then override any desired configs/arguments with the syntax
``{config}={name_of_config_file}`` or ``{config}.{attribute}={value}``
(e.g.``seed=42`` (defined in the main config class), ``backbone=iw/rn50``,
``alglr.=1.e-5``).


## Citation
```
@article{bartlett2022okapi,
  title={Okapi: Generalising Better by Making Statistical Matches Match},
  author={Bartlett, Myles and Romiti, Sara and Sharmanska, Viktoriia and Quadrianto, Novi},
  journal={Advances in neural information processing systems},
  volume={35},
  year={2022}
}
