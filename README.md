# Capsid expansion

This repository contains a standalone Python 3 script to create models of novel capsid structures with T3 and T4 geometry, starting from known structures with T1 geometry. Here T1/T3/T4 refer to triangulation numbers, as per the Klug-Caspar Theory of viral structure organisation, an in-depth explanation of which you can find in [this open-access review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3767311/).

The short summary is that surface of spherical viruses is composed of triangles, and the complexity of said surface can be captured mathematically as the *triangulation number*. T1 is the simplest and smallest structure, containing only one type of protein, sixty copies of which form an icosahedron (see below). By increasing the amount of triangles it becomes possible to build T3 and T4 structures, which are far larger and more complex, featuring multiple types of proteins.

![Triangulation numbers](Triangulation_numbers.png)

Importantly, this expansion can be done by simply copying and rearranging the components of a T1 virus, which is the purpose of this script.

## Setup

The script was developed with Python 3.8.19, Pandas 1.2.4, PyMol 2.4.1, SciPy 1.6.2, Matplotlib 3.3.4, NumPy 1.20.1, and Boltons 23.0.0. It has not been tested for version compatibility, so it’s recommended to create a virtual environment with these specific versions to avoid issues. PyRosetta is an optional library, used for filling in missing atoms (e.g. lysine sidechains that were poorly resolved in the starting T1 crystal structure). The code is compatible with both Windows and Linux.

## Usage

The primary function users should be interacting with is *run_pipeline*. It requires two arguments, the first being a list of four-character codes corresponding to structures of T1 capsids from the Protein Data Bank (PDB) database, e.g. ["4Y5Z"]. The second is a list of integers, containing either 3, 4, or both. These are the triangulation numbers to which the provided T1 capsids will be expanded. Running the script as provided should generate the structures shown below, which can also be found in the [Examples](https://github.com/TomaszKaminski-netizen/capsid-expansion/blob/master/Examples) folder.

![4Y5Z capsids](4Y5Z_capsids.png)

## Contributing

Currently not open to contributions, but comments with suggestions are welcome. This repository is being maintained by [TomaszKaminski-netizen](https://github.com/TomaszKaminski-netizen).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/TomaszKaminski-netizen/capsid-expansion/blob/master/LICENSE.txt) file for details.
