import setuptools

with open("README", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="CoLoRe_corrf_analysis",
    version="0.1",
    author="César Ramírez",
    author_email="cramirez@ifae.es",
    description="Run corrf for 3D clustering. Analyze output from corrf. Make theoretical models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cramirezpe/modules-corrf-analysis/",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'CoLoRe_corrf_search_sims = CoLoRe_corrf_analysis.scripts.available_sims:main',
            'CoLoRe_corrf_run_correlations = CoLoRe_corrf_analysis.scripts.run_correlations:main',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.5',
)