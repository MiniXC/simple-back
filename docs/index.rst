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

.. warning::
    **simple-back** is under active development.
    Any update could introduce breaking changes or even bugs (although we try to avoid the latter as much as possible).
    As soon as the API is finalized, version 1.0 will be released and this disclaimer will be removed.
    
    To get a sense of how far we are along, you can have a look at the `1.0 milestone`_.

.. _1.0 milestone:
   https://github.com/MiniXC/simple-back/milestone/1

Getting Started
===============

:doc:`intro/quickstart`
-----------------------
Build and test a simple strategy using simple-back.

:doc:`intro/debugging`
----------------------
Visualize and improve your strategy.

:doc:`intro/slippage`
---------------------
Move from stateless iterators to stateful strategy objects and configure slippage.

:doc:`intro/example`
--------------------
Just copy example code and get started yourself.

.. toctree::
   :maxdepth: 5
   :caption: Getting Started
   :hidden:

   intro/quickstart
   intro/debugging
   intro/slippage
   intro/example

.. toctree::
   :maxdepth: 5
   :caption: Advanced
   :hidden:

   adv/data_sources

..
   intro/strategies
   intro/data

.. toctree::
   :maxdepth: 5
   :caption: API
   :hidden:

   api/simple_back