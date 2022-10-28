# Module to save logs, associated with results of a certain test (or several)

def decorator(func):
    printer = func
    def wrapped(*args, **kw):
        return printer(*args, **kw, flush=True)
    return wrapped

print = decorator(print)