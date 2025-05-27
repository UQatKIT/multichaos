# Multichaos

Multichaos is a Python package for constructing **Polynomial Chaos Expansions (PCEs)** to approximate response surfaces, especially when dealing with expensive numerical simulations. It supports **multilevel approximation** strategies to balance computational cost and accuracy using an **optimal least squares** approach enhanced with **importance sampling**.

For the underlying theory, refer to the [original publication](https://www.esaim-m2an.org/articles/m2an/abs/2020/02/m2an170180/m2an170180.html).

## Installation

To install, you can `cd` into the multichaos directory and run
```
pip install .
```

## Documentation

The [documentation](https://uqatkit.github.io/multichaos/) provides further information regarding usage, theoretical background and API.

## Citation

This code was used in the course for the following [work](https://www.sciencedirect.com/science/article/pii/S0266892025000785):

```
@article{Salatovic2025,
  title = {Reliable uncertainty quantification for fiber orientation in composite molding processes using multilevel polynomial surrogates},
  ISSN = {0266-8920},
  DOI = {10.1016/j.probengmech.2025.103806},
  journal = {Probabilistic Engineering Mechanics},
  publisher = {Elsevier BV},
  author = {Salatovic,  Stjepan and Krumscheid,  Sebastian and Wittemann,  Florian and K\"{a}rger,  Luise},
  year = {2025},
  month = jul,
  pages = {103806}
}
```