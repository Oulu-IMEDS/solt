# Contribution Guidelines

## Setting up the environment
You need to make sure that you have the right setup. Install anaconda, and you can use the following script
to setup your environment:

```
conda create -q -y -n solt_test_env python=3.8
conda install -y -n solt_test_env pandoc
conda install -y -n solt_test_env pytorch torchvision cpuonly -c pytorch
source activate solt_test_env
conda install -y pip
pip install -r ci/requirements.txt
pip install pre-commit
pre-commit install
pip install -e .
```

## Contributing

1. Implement the augmentation
2. Write tests
3. Make sure that your codecoverage is 100%. The criteria for merging into master is
that all of the tests pass and that the codecoverage is 100%.
