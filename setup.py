from setuptools import setup

setup(
    name='lm',
    packages=['lm'],
    install_requires=[
        'fire',
        'numpy',
        'sentencepiece',
        'torch',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'sp-train = lm.data:sp_train',
            'sp-encode = lm.data:sp_encode',
        ],
    }
)
