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

This code was used in the course for the following work:

```
@misc{salatovic2025reliableuncertaintyquantificationfiber,
      title={Reliable Uncertainty Quantification for Fiber Orientation in Composite Molding Processes using Multilevel Polynomial Surrogates}, 
      author={Stjepan Salatovic and Sebastian Krumscheid and Florian Wittemann and Luise Kärger},
      year={2025},
      eprint={2412.08459},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
}
```