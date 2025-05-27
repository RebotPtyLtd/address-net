from setuptools import setup

setup(
    name='address-net',
    version='1.0',
    packages=['addressnet'],
    url='https://github.com/jasonrig/address-net',
    license='MIT',
    author='Jason Rigby',
    author_email='hello@jasonrig.by',
    description='Splits Australian addresses into their components',
    extras_require={
        "tf": ["tensorflow>=2.0,<3.0", "tensorflow-estimator>=2.0,<3.0"],
        "tf_gpu": ["tensorflow-gpu>=2.0,<3.0", "tensorflow-estimator>=2.0,<3.0"],
    },
    install_requires=[
        'numpy',
        'textdistance'
    ],
    include_package_data=True
)
