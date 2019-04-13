"""
Helpers methods for interacting with python fire.
https://gist.github.com/trhodeos/5a20b438480c880f7e15f08987bd9c0f
adjusted for keyword only arguments support
"""
import functools
import inspect


def only_allow_defined_args(function_to_decorate):
    """
    Decorator which only allows arguments defined to be used.
    Note, we need to specify this, as Fire allows method chaining. This means
    that extra kwargs are kept around and passed to future methods that are
    called. We don't need this, and should fail early if this happens.
    Args:
    function_to_decorate: Function which to decorate.
    Returns:
    Wrapped function.
    """

    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        """Internal wrapper function."""
        valid_names = get_defined_args(function_to_decorate)
        for arg_name in kwargs:
            if arg_name not in valid_names:
                raise ValueError("Unknown argument seen '%s', expected: [%s]" %
                                 (arg_name, ", ".join(valid_names)))
        return function_to_decorate(*args, **kwargs)

    return _return_wrapped


def get_defined_args(function):
    argspec = inspect.getfullargspec(function)
    valid_names = set(argspec.args + argspec.kwonlyargs)
    if 'self' in valid_names:
        valid_names.remove('self')
    return valid_names
