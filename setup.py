from setuptools import setup, find_packages

setup(
    name='PyLauncherMPI',
    version='0.1.0',
    author='John Vouvakis Manousakis',
    author_email='ioannis_vm@berkeley.edu',
    description='A simple MPI-based task scheduler for dynamically distributing commands across MPI processes.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ioannis-vm/PyLauncherMPI',
    packages=find_packages(),
    install_requires=[
        'mpi4py',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'pylaunchermpi=PyLauncherMPI.main:main',  # Adjust the path as necessary
        ],
    },
)
