#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension

from Cython.Build import cythonize
import numpy as np


def main():
    ext_modules = [
        Extension("logiqa.logiqa", ["logiqa/logiqa.pyx"], include_dirs=[np.get_include()])
    ]

    setup(name='LOGIQA',
          description='Long-range genome interactions quality assessment',
          author='Matthias Blum',
          author_email='mat.blum@gmail.com',
          url='http://ngs-qc.org/logiqa/',
          version='1.0',
          zip_safe=False,
          scripts=['bin/logiqa'],
          include_package_data=True,
          package_dir={'logiqa': 'logiqa'},
          packages=['logiqa'],
          install_requires=['Cython>=0.22', 'h5py>=2.5.0', 'numpy>=1.6', 'scipy>=0.15.1'],
          ext_modules=cythonize(ext_modules)
          )


if __name__ == '__main__':
    main()
