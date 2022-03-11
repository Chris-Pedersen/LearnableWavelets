# Wavelets
Applying learnable scattering networks to CAMELs fields and compare inference performance with standard CNN. Uses `pytorch` and `kymatio`, as of March 8th 2022 this requires my own `kymatio` branch at https://github.com/Chris-Pedersen/kymatio, due to padding issues in the standard library.

Most of the `models` code is lifted from https://github.com/bentherien/ParametricScatteringNetworks/, and most of the CAMELs code is lifted from https://camels-multifield-dataset.readthedocs.io/en/latest/.

Will require you to set the path to the CAMELs dataset as an environment variable `export CAMELS_PATH="/path/to/CAMELS/data"`
