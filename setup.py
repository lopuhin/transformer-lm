from setuptools import setup, find_packages

setup(
    version='0.1.0',
    name='lm',
    packages=find_packages(),
    install_requires=[
        # only minimal inference requirements listed
        'attrs',
        'numpy',
        'sentencepiece',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'sp-train = lm.data:sp_train',
            'char-train = lm.data:char_train',
            'tokenize-corpus = lm.data:tokenize_corpus',
            'gpt-2-tf-train = lm.gpt_2_tf.train:main',
            'gpt-2 = lm.main:fire_main',
            'gpt-2-gen = lm.generate:fire_gen_main',
            'lm-web-ui = lm_web_ui.main:main',
        ],
    }
)
