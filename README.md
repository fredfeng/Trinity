# Trinity+: Next-generation Synthesizer for Data Science

Dev Environment Setup
=====================
- Prerequisite:
    - python 3.6+  
- It is preferable to have a dedicated virtualenv for this project:
```
    $ git clone <this repo>
    $ cd Tyrell
    $ mkdir venv
    $ python3 -m venv venv
    $ source venv/bin/activate
```
- Make an editable install with `pip`. This would automatically handles package dependencies. One of our dependency, `z3-solver`, takes a long time to build. Please be patient.
```
    $ pip install -e ".[dev]"
    $ python setup.py sdist  # for package
```
- Test whether the installation is successful
```
    $ parse-tyrell-spec example/toy.tyrell
```
- Run all unit tests
```
    $ python -m unittest discover .
```
- Create a distribution tarball
```
    $ python setup.py sdist
```
  Tarball will be available at `dist/tyrell-<version>.tar.gz`
- Build HTML documentations
```
    $ cd docs
    $ make html
```
  Documentations will be available at `docs/_build/html/index.html`
    
References
- Jia Chen, Ruben Martins, Yanju Chen, Yu Feng, Isil Dillig. Trinity: An Extensible Synthesis Framework for Data Science. PVLDB'19.
- Yu Feng, Ruben Martins, Osbert Bastani, Isil Dillig. Program Synthesis using Conflict-Driven Learning. PLDI'18.
- Yu Feng, Ruben Martins, Jacob Van Geffen, Isil Dillig, Swarat Chaudhuri. Component-based Synthesis of Table Consolidation and Transformation Tasks from Examples. PLDI'17
- Yu Feng, Ruben Martins, Yuepeng Wang, Isil Dillig, Thomas W. Reps. Component-Based Synthesis for Complex APIs. POPL'17
