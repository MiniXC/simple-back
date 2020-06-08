=====================
simple-back |version|
=====================

|Build Status| |PyPi| |Code Style| |codecov|

    .. |Build Status| image:: https://github.com/MiniXC/simple-back/workflows/build/badge.svg
        :target: https://github.com/MiniXC/simple-back

    .. |PyPi| image:: https://img.shields.io/pypi/v/simple-back
        :target: https://pypi.org/project/simple-back/

    .. |Code Style| image:: https://img.shields.io/badge/code%20style-black-000000.svg

    .. |codecov| image:: https://codecov.io/gh/MiniXC/simple-back/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/MiniXC/simple-back

**simple-back** is a backtester providing easy ways to test trading
strategies, with minimal amounts of code, while avoiding time leaks.
At the same time, we aim to make it easy to use your own data and price sources.

This package is especially useful when used in Jupyter Notebooks,
as it provides ways to show you live feedback
while your backtests are being run.

Getting Started
===============

.. toctree::
   :caption: Getting Started
   :hidden:

   intro/quickstart
   intro/strategies
   intro/data
   intro/example


:doc:`intro/quickstart`
-----------------------
Build and test a simple strategy using simple-back.

:doc:`intro/strategies`
-----------------------
Create a :class:`~simple_back.strategies.Strategy` object and run multiple strategies at the same time.

:doc:`intro/data`
-----------------
Write your first :class:`~simple_back.data_provider.DataProvider`

:doc:`intro/example`
--------------------
Just copy example code and get started yourself.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`