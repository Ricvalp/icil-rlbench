from setuptools import find_namespace_packages, setup


core_requirements = [
    "numpy",
    "torch",
    "absl-py",
    "ml-collections",
    "h5py",
    "diffusers",
    "wandb",
]


extras_require = {
    "wandb": [
        "wandb",
        "matplotlib",
    ],
    "inspect": [
        "plotly",
        "matplotlib",
    ],
    "profile": [
        "tqdm",
    ],
    "dev": [
        "pytest",
    ],
}


setup(
    name="icil-pretrain",
    version="0.1.0",
    description="ICIL Perceiver pretraining on cached RLBench dense H5 data.",
    packages=find_namespace_packages(include=["icil*"]),
    include_package_data=True,
    install_requires=core_requirements,
    extras_require=extras_require,
    python_requires=">=3.9",
)
