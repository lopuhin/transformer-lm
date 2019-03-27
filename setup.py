from setuptools import setup, find_packages

setup(
    name='lm',
    packages=find_packages(),
    install_requires=[
        'attrs',
        'fire',
        'numpy',
        'sentencepiece',
        'tensorflow',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'sp-train = lm.data:sp_train',
            'sp-encode = lm.data:sp_encode',
            'gpt-2-tf-train = lm.gpt_2_tf.train:main',
            'gpt-2-tf2 = lm.gpt_2_tf2.main:fire_main',
        ],
    }
)
