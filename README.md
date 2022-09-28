# Wavelets
Learnable scattering networks.


Code used in the ICML Machine learning for astronomy workshop paper "Learnable Wavelet Neural Networks for Cosmological Inference" (https://ml4astro.github.io/icml2022/assets/40.pdf). The version of the code used to generate results for this paper is here, and requires a custom version of the scattering transform code `kymatio` at https://github.com/Chris-Pedersen/kymatio, due to padding issues in the standard library.

Most of the `models` code is lifted from https://github.com/bentherien/ParametricScatteringNetworks/, and most of the CAMELs code is lifted from https://camels-multifield-dataset.readthedocs.io/en/latest/.

Will require you to set the path to the CAMELs dataset as an environment variable `export CAMELS_PATH="/path/to/CAMELS/data"`
