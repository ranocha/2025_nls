# High-order mass- and energy-conserving methods for the nonlinear Schrödinger equation and its hyperbolization

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17361026.svg)](https://zenodo.org/doi/10.5281/zenodo.17361026)

This repository contains information and code to reproduce the results presented
in the article
```bibtex
@online{ranocha2025high,
  title={High-order mass- and energy-conserving methods for the
         nonlinear {S}chrödinger equation and its hyperbolization},
  author={Ranocha, Hendrik and Ketcheson, David I},
  year={2025},
  month={10},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{ranocha2025highRepro,
  title={Reproducibility repository for
         "{H}igh-order mass- and energy-conserving methods for the
         nonlinear {S}chrödinger equation and its hyperbolization"},
  author={Ranocha, Hendrik and Ketcheson, David I},
  year={2025},
  howpublished={\url{https://github.com/ranocha/2025_nls}},
  doi={10.5281/zenodo.17361026}
}
```

## Abstract

We propose a class of numerical methods for the nonlinear Schrödinger (NLS) equation that conserves mass and energy, is of arbitrarily high-order accuracy in space and time, and requires only the solution of a scalar algebraic equation per time step.  We show that some existing spatial discretizations, including the popular Fourier spectral method, are in fact energy-conserving if one considers the appropriate form of the energy density. We develop a new relaxation-type approach for conserving multiple nonlinear functionals that is more efficient and robust for the NLS equation compared to the existing multiple-relaxation approach. The accuracy and efficiency of the new schemes is demonstrated on test problems for both the focusing and defocusing NLS.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/).
The numerical experiments presented in this article were performed using
Julia v1.10.10.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory of this repository and follow the instructions
described in the `README.md` file therein.


## Authors

- [Hendrik Ranocha](https://ranocha.de) (Johannes Gutenberg University Mainz, Germany)
- David I. Ketcheson (KAUST, Saudi Arabia)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
