
# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = ''

setup(
    long_description=readme,
    name='simple-back',
    version='0.6.0',
    description='A backtester with minimal setup and sensible defaults.',
    python_requires='==3.*,>=3.6.9',
    author='Christoph Minixhofer',
    author_email='christoph.minixhofer@gmail.com',
    packages=['simple_back'],
    package_dir={"": "."},
    package_data={},
    install_requires=['beautifulsoup4==4.*,>=4.9.1', 'diskcache==4.*,>=4.1.0', 'memoization==0.*,>=0.3.1', 'numpy==1.*,>=1.18.4', 'pandas==1.*,>=1.0.3', 'pandas-market-calendars==1.*,>=1.3.5', 'pytz==2020.*,>=2020.1.0', 'requests-html==0.*,>=0.10.0', 'tabulate==0.*,>=0.8.7', 'yahoo-fin==0.*,>=0.8.5'],
    dependency_links=['git+https://github.com/PouncySilverkitten/pylint-badge.git#egg=pylint-badge'],
    extras_require={"dev": ["black==19.*,>=19.10.0.b0", "dephell==0.*,>=0.8.3", "flake8==3.*,>=3.8.2", "ipython==7.*,>=7.15.0", "nbsphinx==0.*,>=0.7.0", "pandoc==1.*,>=1.0.2", "pylint-badge", "pytest==5.*,>=5.2.0", "pytest-cov==2.*,>=2.9.0", "sphinx==3.*,>=3.0.4", "sphinx-autodoc-typehints==1.*,>=1.10.3", "sphinx-copybutton==0.*,>=0.2.11", "sphinx-rtd-theme==0.*,>=0.4.3"]},
)
