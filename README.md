# Learnable Wavelet Networks
Scattering convolutional networks with learnable filter parameters implemented in PyTorch.

### Set up environment in `conda`
If using conda, set up an environment in the following way:

`conda env create -f environment.yml`

Once installed, activate the environment:

`conda activate wavelet`

and install the `learnable_wavelet` package via

`pip install .`

NB to run the `playground/example_notebook.ipynb`, first run the following to download the small sample of CAMELs data:

`wget -O test_data/maps_Mtot_1k.npy "https://www.dropbox.com/s/oqy3h3l4cdgfdeq/maps_Mtot_1k.npy?dl=0"`

Code used in the ICML 2022 Workshop on Machine Learning for Astrophysics workshop paper ["Learnable Wavelet Neural Networks for Cosmological Inference"](https://ml4astro.github.io/icml2022/assets/40.pdf). The code has been significantly refactored since this paper but produces consistent results. Nevertheless for completeness, the specific version of the code used to generate results for this paper is [here](https://github.com/Chris-Pedersen/LearnableWavelets/tree/c71b723ce4de99e7c272e8da7d615f4d9edebb54), and requires a custom version of the scattering transform code `kymatio` at [found here](https://github.com/Chris-Pedersen/kymatio) due to padding issues in the standard library. 

Most of the `models` code is lifted from https://github.com/bentherien/ParametricScatteringNetworks/, and most of the CAMELs code is lifted from https://camels-multifield-dataset.readthedocs.io/en/latest/.

