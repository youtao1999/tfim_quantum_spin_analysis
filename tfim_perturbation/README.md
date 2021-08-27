# tfimED

A Python numerical exact diagonalization code for [transverse field Ising models](https://en.wikipedia.org/wiki/Ising_model) 

This allows you to compute ground state and low energy properties of Hamiltonians of the form:

H = -\sum_{ij} J_{ij}\sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i

Tested with Python 2.7.11.


## Requirements

* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [Progressbar](https://pypi.python.org/pypi/progressbar2)


## Examples

* `python tfim_diag.py --help`
* `python tfim_diag.py 8`
* `python tfim_diag.py --full --h_min 0.5 --h_max 4 --dh 0.5 4 -o myoutputfile`
* `python tfim_diag.py -D 2 --h_max 6 2`
* `python tfim_diag.py --load my_matrix_filename_base`

- - - -

* `python tfim_build.py -D 2 4 -o my_matrix_filename_base`
