from setuptools import setup, find_packages

setup(
    name='it4851_nhom13',
    version='1.0',
    description='Animal Image Classification',
    author='minhdao',
    packages=find_packages(exclude=[
        'docs', 'tests', 'static', 'templates', '.gitignore', 'README.md', 'data'
    ]),
    install_requires=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'jupyterlab',
        'pillow',
        'pylint',
        'autopep8',
        'rope',
        'flask',
        'pymongo',
        'python-dotenv',
        'opencv-python'
    ],
)
