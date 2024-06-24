from setuptools import setup, find_packages, Extension


setup(name='pygdsdesign',
      version='0.16.0',
      use_2to3=False,
      author='Étienne Dumur, Sacha Wos',
      author_email='etienne.dumur@gmail.com',
      maintainer='Étienne Dumur',
      maintainer_email='etienne.dumur@gmail.com',
      description='pygdsdesign provides some function to more efficiently create gds files.',
      long_description=open('README.md', encoding='utf8').read(),
      classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering'
      ],
      license='BSL',
      packages=find_packages(),
      ext_modules = [Extension("clipper", ["pygdsdesign/clipper/clipper.cpp"])],
      install_requires=[
          'numpy',
          'pytest',
          'scipy',
          'tqdm',
          'typing_extensions',
      ],
      )
