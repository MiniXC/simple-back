from io import StringIO
import sys
import os
from IPython.display import clear_output


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def _cls():
    clear_output(wait=True)
    os.system("cls" if os.name == "nt" else "clear")
