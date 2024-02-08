import setuptools

with open("README.md", encoding = 'utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'dfxm_fwrd_sim',
    version = '0.0.0',
    author = 'Mads Carlsen',
    author_email = 'madsac@dtu.dk',
    description = 'Simulation of DFXM experiments with coherent wavefron techniques',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages = ['dfxm_fwrd_sim'],
    python_requires = '>=3.6',
)