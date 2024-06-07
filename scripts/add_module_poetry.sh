# Description: Add a module to the poetry project
# Usage: sh scripts/add_module_poetry.sh <module_name>


PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring poetry add $1