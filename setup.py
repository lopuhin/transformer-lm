from setuptools import setup

setup(
    name='lm',
    packages=['lm'],
    install_requires=[
        'numpy',
        'sentencepiece',
        'torch',
    ],
)
