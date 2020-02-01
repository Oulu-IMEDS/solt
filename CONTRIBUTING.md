# Contribution Guidelines

## Setting up the environment
1. Create a fresh conda env and install the dependencies from `ci/requirements.txt`.
2. Install `pre-commit` from pip
3. Install the library as `pip install -e .`
4. Install pre-commit hooks as `pre-commit` install

## Contributing

1. Implement the augmentation
2. Write tests
3. Make sure that your codecoverage is 100%. The criteria for merging into master is
that all of the tests pass and that the codecoverage is 100%.