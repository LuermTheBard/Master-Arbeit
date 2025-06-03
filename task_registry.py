registered_tasks = {}


def task(func):
    registered_tasks[func.__name__] = func
    return func
