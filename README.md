# Learnable Wavelet Networks
Scattering convolutional networks with learnable filter parameters implemented in PyTorch.


Code used in the ICML 2022 Workshop on Machine Learning for Astrophysics workshop paper ["Learnable Wavelet Neural Networks for Cosmological Inference"](https://ml4astro.github.io/icml2022/assets/40.pdf). The code has been significantly refactored since this paper but produces consistent results. Nevertheless for completeness, the specific version of the code used to generate results for this paper is [here](https://github.com/Chris-Pedersen/LearnableWavelets/tree/c71b723ce4de99e7c272e8da7d615f4d9edebb54), and requires a custom version of the scattering transform code `kymatio` at [found here](https://github.com/Chris-Pedersen/kymatio) due to padding issues in the standard library. 

Most of the `models` code is lifted from https://github.com/bentherien/ParametricScatteringNetworks/, and most of the CAMELs code is lifted from https://camels-multifield-dataset.readthedocs.io/en/latest/.

Will require you to set the path to the CAMELs dataset as an environment variable `export CAMELS_PATH="/path/to/CAMELS/data"`
