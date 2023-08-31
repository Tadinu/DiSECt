from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'DiSECt package'

# Setting up
setup(
       # the name must match the folder name 'disect'
        name="disect", 
        version=VERSION,
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
)