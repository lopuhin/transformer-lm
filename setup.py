from setuptools import setup

setup(
    name='lm',
    packages=['lm'],
    install_requires=[
        'numpy',
        'sentencepiece',
        'torch',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'lm-train = lm.train:main',
            'sp-train = lm.data:sp_train',
        ],
    }
)
