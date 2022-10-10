from importlib.machinery import SourceFileLoader

from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

version = SourceFileLoader('musicsections.version', 'musicsections/version.py').load_module()

setup(
   name='musicsections',
   version=version.version,
   description='Music segmentation using deep embeddings and section fusion',
   author='Justin Salamon, Oriol Nieto, Nicholas J. Bryan',
   url='https://github.com/justinsalamon/musicseg_deepemb',
   packages=['musicsections'],
   long_description=long_description,
   long_description_content_type='text/markdown',
   keywords='Music segmentation deep embeddings section fusion',
   license='See LICENSE.txt',
   install_requires=[
      'torch==1.1.0',
      'torchvision==0.2.1',
      'torchsummary==1.5.1',
      'numpy==1.19.2',
      'scipy==1.4.1',
      'scikit-learn==0.22.1',
      'audioread==2.1.8',
      'numba==0.48.0',
      'pandas==1.0.3',
      'resampy==0.2.2',
      'SoundFile==0.10.3.post1',
      'sox==1.3.7',
      'h5py==3.1.0',
      'librosa==0.7.0',
      'tensorflow==2.6.0',
      'keras==2.6.0',
      'mir_eval==0.6',
      'madmom==0.16.1',
      'matplotlib'
   ],
)
