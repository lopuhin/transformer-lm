from setuptools import setup, find_packages

setup(
    name='lm',
    packages=find_packages(),
    install_requires=[
        'attrs',
        'json-log-plots',
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
            'gpt-2-tf-train = lm.gpt_2_tf.train:main',
            'gpt-2 = lm.main:fire_main',
        ],
    }
)
