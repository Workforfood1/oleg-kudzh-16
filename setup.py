from setuptools import find_packages, setup

setup(
    name='mlops_project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'pydantic',
        'pytest',
        'pylint'
    ],
)
