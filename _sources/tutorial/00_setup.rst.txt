====================
Setting up a project
====================

First of all, make sure that your system has a python version >=3.6 installed. 

It is recommended that a `virtual environment <https://docs.python.org/3/library/venv.html>`_ is used when starting with a new project:

  .. code-block:: bash

   $ mkdir venv
   $ python3 -m venv ./venv
   $ source venv/bin/activate

Now, there are several ways you may install the Tyrell framework and its dependencies. 

- Install Tyrell from PyPI:

  .. warning:: This option is not possible at the moment.

  .. code-block:: bash

   $ pip install tyrell

- Obtaining a tarball distribution of tyrell. Suppose the name of the tarball is ``tyrell-0.1.tar.gz``:

  .. code-block:: bash

   $ pip install tyrell-0.1.tar.gz


.. note:: One of Tyrell's dependency, `z3-solver`, takes a long time to build. Please be patient.

To test whether the installation is successful, run the following command:

  .. code-block:: bash

   $ parse-tyrell-spec --help

If the help message is correctly shown, everything should be good. 