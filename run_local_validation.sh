#!/bin/bash

mypy . && echo "mypy passed!"
flake8 && echo "flake8 passed!"
pytest && echo "pytest passed!"
echo "All validation passed!"