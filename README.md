# pygdsdesign

Pydesign is a Python library designed to generate GDS files with Python scripts. The library's is built upon code from GDSpy, a popular open-source toolkit for layout design and manipulation. Moreover, Pydesign use the power of Clipper, to execute precise boolean operations. 

## ⚙️ Installation

Since Pydesign incorporates C++ code that requires compilation (Clipper), ensure that your development environment supports compilation. 
For instance, you can use Microsoft Visual C++ for Windows.

To install pygdsdesign, simply run:
`pip install .`

(You can also add the `-e` flag if you want to install the library in editable mode.)

After installing the library, it is recommended to verify the installation to ensure that all the code functions correctly. To do this, use pytest and run the tests provided in the examples folder.

## 🖼️ Gallery

Example of what you can do with pygdsdesign
### Spiral resonator

![spiral resonator](examples/spiral_resonator.png "spiral resonator")

### CPW resonators

![cpw resonators](examples/cpw_resonators.png "cpw resonators")


## 🛠️ How to start

You can find many scripts in the `Examples/` folder showing different user cases.


## Acknowledgment

This library is authored by Etienne Dumur and incorporates significant code from GDSPY, a library written by Lucas H. Gabrielli. For boolean operations, pygdsdesign utilizes Clipper, a library developed by Angus Johnson. Furthermore, the library has been ported from Python 3.8 to 3.11 and beyond by Sacha Wos.
