import sys
import subprocess

from handle_data.handle_data import create_1d_correlation_plot_data
from import_data.import_data import import_1d_correlation_data
from plot_data.plot_data import plot_1d_correlations

# Dictionary to store registered tasks
registered_tasks = {}


# Decorator to register functions as tasks
def task(func):
    registered_tasks[func.__name__] = func
    return func


@task
def dummy_task():
    subprocess.run(["python", "-m", "scrap.scripts.dummy_script", "calc_2_plus_2"], check=True)


@task
def plot_all_1d_corr():
    """
    Plots all 1D correlations.
    """
    one_dim_correlation_data = import_1d_correlation_data()
    one_dim_correlation_plot_data = create_1d_correlation_plot_data(one_dim_correlation_data)
    plot_1d_correlations(one_dim_correlation_plot_data)

@task
def plot_line_1d_corr(line_name=None):
    """
    Plots 1D correlations for a specific line.

    Args:
        line_name (str): The name of the line to plot.

    Raises:
        ValueError: If line_name is not provided.
    """
    # Prüfen, ob line_name definiert ist
    if not line_name:
        raise ValueError("Please specify a line name in the following form: plot_line_1d_corr:line_name")

    one_dim_correlation_data = import_1d_correlation_data()
    one_dim_correlation_plot_data = create_1d_correlation_plot_data(one_dim_correlation_data)
    plot_1d_correlations(one_dim_correlation_plot_data, line_name=line_name)


def run_task(tasks):
    for task in tasks:
        try:
            # Split task name and additional arguments
            parts = task.split(":")
            task_name = parts[0]
            task_args = parts[1:] if len(parts) > 1 else []

            print(f"Running {task_name} with arguments {task_args}...")

            # Call the task with unpacked arguments
            registered_tasks[task_name](*task_args)
        except KeyError:
            print(f"Task '{task}' is not available.")
        except Exception as e:
            print(f"An error occurred while running '{task}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one task name, or use 'list_tasks' to see all available tasks.")
    elif sys.argv[1] == "list_tasks":
        print("Available tasks:")
        for task_name in registered_tasks:
            print(f"- {task_name}")
    else:
        run_task(sys.argv[1:])
