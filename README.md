# Tyrell+: Next-generation Synthesizer for Data Science

Dev Environment Setup
=====================
- Prerequisite:
    - python 3.5+  
- It is preferable to have a dedicated virtualenv for this project:
```
    $ git clone <this repo>
    $ cd Tyrell
    $ mkdir venv
    $ python3 -m venv venv
    $ source venv/bin/activate
```
- Make an editable install with `pip`. This would automatically handles package dependencies.
```
    $ pip install -e .
```
- Test whether the installation is successful
```
    $ parse-spec example/toy.tyrell
```
- Run all unit tests
```
    $ python -m unittest discover tests
```
    
Morpheus prerequisite
=====================
- R 3.3.3 (rpy2 won't work with the latest R)
    - dplyr: install.packages("dplyr")
    - tidyr: install.packages("tidyr")
    - devtools: install.packages("devtools")
- rpy2: pip3 install rpy2
- Install MorpheusData for PLDI17 (optional): devtools::install_github("fredfeng/MorpheusData")
- Load MorpheusData for PLDI17 in R (optional): library(MorpheusData)
