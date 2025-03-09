# Title: Organizing Project Repo Files into a Standard Subfolder Structure.

The goal of this exercise is to reorganize the files into a more-standard structure,
adding a new build file type called `pyproject.toml` to make it so that Python scripts can find the functions we make in this project, no matter what folder the scripts are stored in.

```
<project_name>
|
├── data/
|   └── data.nc
|
├── results/
|       └── PSTHs.png
|
├── scripts/
|   └── calculate_psths.py
|
├── src/
|   ├── project_name/
|   |   ├── __init__.py
|   |   └── utils.py
|   |
|   └── <module>.py
|    
├── docs/
|   ├── exercise1.md
|   └── exercise2.md
|
├── pyproject.toml
├── environment.yml
└── README.md
```


## Reference: A Minimal `pyproject.toml` File.

The `pyproject.toml` file contains installation instructions for Python projects, so that installing the functions is as simple as `pip install -e .`

Here is the absolute minimum needed for the project to work:
```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "package_name"
version = "v0.0.1"
```

| **`Command`** | **`Description`** |
| :-- | :-- |
| `pip install -e .` | Install the packages and its dependencies into the current python environment, but keep it easy to modify the files. |
| `pip uninstall .` | Remove this package from the current python environment.  Note: won't uninstall the dependencies. |

## Instructions

1. Refactor Functions out to Modules: 
    1. Move the functions you created in Exercise 1 to a module called `utils.py`, right next to the script.
    2. Modify the script so that they import the functions from the module.
    3. Make sure the script still runs.
    4. Make a git commit to save the success.
2. Refactor the Project into the subfolders mentioned in the reference above.
    1. Move the files into folders.
    2. The script should not work anymore, since the functions should be in `src/psth/` and the script should be in `scripts/`, making it hard for the script to find the functions.
    3. Write a minimal `pyproject.toml` file using the reference above and use it to install the `psth` package into the project. 
    4. Modify the script to import from `psth` to get the functions.  Now, the script should work again.
    5. Make a Git Commit to save the success.


