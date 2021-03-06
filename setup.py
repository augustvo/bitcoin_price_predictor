from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='bitcoin_price_predictor',
      version="1.0",
      description="Bitcoin price prediction with an Deep Learning RNN, based on other stock markets",
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/bitcoin_price_predictor-run'],
      zip_safe=False)
