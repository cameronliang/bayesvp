import io
from setuptools import setup

with io.open('README.rst', encoding = 'utf-8') as f:
	long_description = f.read()

setup(name='bayesvp',
    version='0.2.2',
    description='Bayesian MCMC Voigt Profile Fitting',
    long_description = long_description,
    url='https://github.com/cameronliang/bayesvp',
    author='Cameron Liang',
    author_email='cameron.liang@gmail.com',
    license='MIT',
    packages=['bayesvp'],
    include_package_data=True,
    entry_points={
        'console_scripts': ['bvpfit = bayesvp.scripts.bvpfit:main',
        'bvp_write_config = bayesvp.scripts.bvp_write_config:main',
        'bvp_process_model = bayesvp.scripts.bvp_process_model:main'],
    },
    install_requires=[
        'numpy', 'scipy',
        'matplotlib', 
        'kombine','emcee'
        ],
    zip_safe=False)
