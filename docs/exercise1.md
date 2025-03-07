
# Title: Refactoring Pipeline Steps into Functions

In the `calculate_psths.py` script, electrophysiological recordings from an experiment reported in [Steinmetz et al, 2019]
is downloaded and plotted as PSTHs broken down by brain area and stimulus intensity. While this is a 
fairly common analysis, it requires many steps to perform.

The goal of this exercise is to practice breaking down scripts into shorter steps by turning blocks of 
code into functions. This process of changing how code is organized without changing how it behaves is 
called "refactoring".  By refactoring code into organized bits, we can make it easier to understand, 
manage, and modify.


## Python Syntax Reference

Defining Functions in Python uses the following basic syntax:

```python
def function_name(arg1, arg2):
    value = arg1 + arg2  # code uses the input arguments.  
    output = value + 1   # variables that are only used in the function will be deleted when the function is finished.
    return output        # Sends a variable to the code calling the function.
```

Calling (i.e. "Running") a Function in Python can use many differeent syntaxes:

```python
val1 = 1
val2 = 2
output = function_name(val1, val2)            # Option 1: Positional Arguments
output = function_name(arg1=val1, arg2=val2)  # Option 2: Keyword Arguments
output = function_name(val1, arg2=val2)       # Option 3: Mixed Positional and Keyword Arguements.
```


## Instructions

1. In the VSCode Terminal, Create a Conda or Pixi environment and install needed dependencies:
    1. Create the sandbox environment from the dependency files:
    - Conda: `conda env create -p ./venv -f environment.yaml`
    - Pixi: `pixi install`
    2. Activate the Environment in your VSCode termainal:
    - Conda: `conda activate ./venv`
    - Pixi: `pixi shell`
    3. Set up the IPykernel config for VSCode's Interactive Mode to Use:
    - `python -m ipykernel install --name psth`

2. Run the `calculate_psths.py` script to make sure it works:
    - `python calculate_psths.py`  (should see new files appear: `data.nc` and `PSTHs.png`)


3. For each code section in the `calculate_psths.py` script:

    1. Refactor the section into two parts: a function and a call to that function, using
    VSCode's Interactive Mode to test that the section stil works with the modified code.  
    2. Each time a function is made successfully, make a git commit to save your progress.
