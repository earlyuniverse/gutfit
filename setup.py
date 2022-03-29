from setuptools import setup, find_packages
setup(
  name = 'gutfit',
  version = '0.1',
  description = 'gutfit',
  url = 'https://github.com/earlyuniverse/gutfit',
  author = 'Holger Schulz, Jessica Turner',
  author_email = 'iamholger@gmail.com,jessicaturner.5390@gmail.com',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'numpy',
    'scipy',
    'matplotlib',
  ],
  python_requires='>3.6.0',
  # scripts=['bin/uls-calc', 'bin/uls-scan', 'bin/uls-nest', 'bin/uls-models'],
  extras_require = {
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
