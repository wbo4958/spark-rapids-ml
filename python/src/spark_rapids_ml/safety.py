import inspect
import functools
from abc import abstractmethod

from spark_rapids_ml import gorilla

_AUTOLOGGING_TEST_MODE_ENV_VAR = "MLFLOW_AUTOLOGGING_TESTING"

_AUTOLOGGING_PATCHES = {}

# Function attribute used for testing purposes to verify that a given function
# has been wrapped with the `exception_safe_function_for_class` and
# `picklable_exception_safe_function` decorators
_ATTRIBUTE_EXCEPTION_SAFE = "exception_safe"


def _safe_function(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as e:
        print("Encountered unexpected error during autologging: %s", e)


def picklable_exception_safe_function(function):
    """
    Wraps the specified function with broad exception handling to guard
    against unexpected errors during autologging while preserving picklability.
    """
    return update_wrapper_extended(functools.partial(_safe_function, function), function)


class PatchFunction:
    """
    Base class representing a function patch implementation with a callback for error handling.
    `PatchFunction` should be subclassed and used in conjunction with `safe_patch` to
    safely modify the implementation of a function. Subclasses of `PatchFunction` should
    use `_patch_implementation` to define modified ("patched") function implementations and
    `_on_exception` to define cleanup logic when `_patch_implementation` terminates due
    to an unhandled exception.
    """

    @abstractmethod
    def _patch_implementation(self, original, *args, **kwargs):
        """
        Invokes the patch function code.

        :param original: The original, underlying function over which the `PatchFunction`
                         is being applied.
        :param *args: The positional arguments passed to the original function.
        :param **kwargs: The keyword arguments passed to the original function.
        """
        pass

    @abstractmethod
    def _on_exception(self, exception):
        """
        Called when an unhandled standard Python exception (i.e. an exception inheriting from
        `Exception`) or a `KeyboardInterrupt` prematurely terminates the execution of
        `_patch_implementation`.

        :param exception: The unhandled exception thrown by `_patch_implementation`.
        """
        pass

    @classmethod
    def call(cls, original, *args, **kwargs):
        return cls().__call__(original, *args, **kwargs)

    def __call__(self, original, *args, **kwargs):
        try:
            return self._patch_implementation(original, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            try:
                self._on_exception(e)
            finally:
                # Regardless of what happens during the `_on_exception` callback, reraise
                # the original implementation exception once the callback completes
                raise e


def safe_patch(
        autologging_integration, destination, function_name, patch_function, manage_run=False
):
    """
    Patches the specified `function_name` on the specified `destination` class for autologging
    purposes, preceding its implementation with an error-safe copy of the specified patch
    `patch_function` with the following error handling behavior:
        - Exceptions thrown from the underlying / original function
          (`<destination>.<function_name>`) are propagated to the caller.
        - Exceptions thrown from other parts of the patched implementation (`patch_function`)
          are caught and logged as warnings.
    :param autologging_integration: The name of the autologging integration associated with the
                                    patch.
    :param destination: The Python class on which the patch is being defined.
    :param function_name: The name of the function to patch on the specified `destination` class.
    :param patch_function: The patched function code to apply. This is either a `PatchFunction`
                           class definition or a function object. If it is a function object, the
                           first argument should be reserved for an `original` method argument
                           representing the underlying / original function. Subsequent arguments
                           should be identical to those of the original function being patched.
    :param manage_run: If `True`, applies the `with_managed_run` wrapper to the specified
                       `patch_function`, which automatically creates & terminates an MLflow
                       active run during patch code execution if necessary. If `False`,
                       does not apply the `with_managed_run` wrapper to the specified
                       `patch_function`.
    """
    patch_is_class = inspect.isclass(patch_function)
    if patch_is_class:
        assert issubclass(patch_function, PatchFunction)
    else:
        assert callable(patch_function)

    original_fn = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=False
    )
    # Retrieve raw attribute while bypassing the descriptor protocol
    raw_original_obj = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=True
    )
    if original_fn != raw_original_obj:
        raise RuntimeError(f"Unsupport patch on {str(destination)}.{function_name}")
    elif isinstance(original_fn, property):
        is_property_method = True

        # For property decorated methods (a kind of method delegation), e.g.
        # class A:
        #   @property
        #   def f1(self):
        #     ...
        #     return delegated_f1
        #
        # suppose `a1` is an instance of class `A`,
        # `A.f1.fget` will get the original `def f1(self)` method,
        # and `A.f1.fget(a1)` will be equivalent to `a1.f1()` and
        # its return value will be the `delegated_f1` function.
        # So using the `property.fget` we can construct the (delegated) "original_fn"
        def original(self, *args, **kwargs):
            # the `original_fn.fget` will get the original method decorated by `property`
            # the `original_fn.fget(self)` will get the delegated function returned by the
            # property decorated method.
            bound_delegate_method = original_fn.fget(self)
            return bound_delegate_method(*args, **kwargs)

    else:
        original = original_fn
        is_property_method = False

    def safe_patch_function(*args, **kwargs):
        """
        A safe wrapper around the specified `patch_function` implementation designed to
        handle exceptions thrown during the execution of `patch_function`. This wrapper
        distinguishes exceptions thrown from the underlying / original function
        (`<destination>.<function_name>`) from exceptions thrown from other parts of
        `patch_function`. This distinction is made by passing an augmented version of the
        underlying / original function to `patch_function` that uses nonlocal state to track
        whether or not it has been executed and whether or not it threw an exception.
        Exceptions thrown from the underlying / original function are propagated to the caller,
        while exceptions thrown from other parts of `patch_function` are caught and logged as
        warnings.
        """
        # Reroute warnings encountered during the patch function implementation to an MLflow event
        # logger, and enforce silent mode if applicable (i.e. if the corresponding autologging
        # integration was called with `silent=True`), hiding MLflow event logging statements and
        # hiding all warnings in the autologging preamble and postamble (i.e. the code surrounding
        # the user's original / underlying ML function). Non-MLflow warnings are enabled during the
        # execution of the original / underlying ML function
        #
        # Note that we've opted *not* to apply this context manager as a decorator on
        # `safe_patch_function` because the context-manager-as-decorator pattern uses
        # `contextlib.ContextDecorator`, which creates generator expressions that cannot be pickled
        # during model serialization by ML frameworks such as scikit-learn
        if False:
            """For some reason, disable the replacement."""
            return original(*args, **kwargs)

        # Whether or not the original / underlying function has been called during the
        # execution of patched code
        original_has_been_called = False
        # The value returned by the call to the original / underlying function during
        # the execution of patched code
        original_result = None
        # Whether or not an exception was raised from within the original / underlying function
        # during the execution of patched code
        failed_during_original = False
        # The active MLflow run (if any) associated with patch code execution
        patch_function_run_for_testing = None
        # The exception raised during executing patching function
        patch_function_exception = None

        def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
            try:
                original_fn_result = original_fn(*og_args, **og_kwargs)
                return original_fn_result
            except Exception as original_fn_e:
                nonlocal failed_during_original
                failed_during_original = True
                raise

        try:

            def call_original(*og_args, **og_kwargs):
                def _original_fn(*_og_args, **_og_kwargs):
                    nonlocal original_has_been_called
                    original_has_been_called = True

                    nonlocal original_result
                    # Show all non-MLflow warnings as normal (i.e. not as event logs)
                    # during original function execution, even if silent mode is enabled
                    # (`silent=True`), since these warnings originate from the ML framework
                    # or one of its dependencies and are likely relevant to the caller

                    original_result = original(*_og_args, **_og_kwargs)
                    return original_result

                return call_original_fn_with_event_logging(_original_fn, og_args, og_kwargs)

            # Apply the name, docstring, and signature of `original` to `call_original`.
            # This is important because several autologging patch implementations inspect
            # the signature of the `original` argument during execution
            call_original = update_wrapper_extended(call_original, original)

            if patch_is_class:
                return patch_function.call(call_original, *args, **kwargs)
            else:
                return patch_function(call_original, *args, **kwargs)


        except Exception as e:
            patch_function_exception = e
            # Exceptions thrown during execution of the original function should be
            # propagated to the caller. Additionally, exceptions encountered during test
            # mode should be reraised to detect bugs in autologging implementations
            if failed_during_original:
                raise

        # try:
        #     if original_has_been_called:
        #         return original_result
        #     else:
        #         return call_original_fn_with_event_logging(original, args, kwargs)
        # finally:
        #     pass

    if is_property_method:
        # Create a patched function (also property decorated)
        # like:
        #
        # class A:
        # @property
        # def get_bound_safe_patch_fn(self):
        #   original_fn.fget(self) # do availability check
        #   return bound_safe_patch_fn
        #
        # Suppose `a1` is instance of class A,
        # then `a1.get_bound_safe_patch_fn(*args, **kwargs)` will be equivalent to
        # `bound_safe_patch_fn(*args, **kwargs)`
        def get_bound_safe_patch_fn(self):
            # This `original_fn.fget` call is for availability check, if it raise error
            # then `hasattr(obj, {func_name})` will return False
            # so it mimic the original property behavior.
            original_fn.fget(self)

            def bound_safe_patch_fn(*args, **kwargs):
                return safe_patch_function(self, *args, **kwargs)

            # Make bound method `instance.target_method` keep the same doc and signature
            bound_safe_patch_fn = update_wrapper_extended(bound_safe_patch_fn, original_fn.fget)
            # Here return the bound safe patch function because user call property decorated
            # method will like `instance.property_decorated_method(...)`, and internally it will
            # call the `bound_safe_patch_fn`, the argument list don't include the `self` argument,
            # so return bound function here.
            return bound_safe_patch_fn

        # Make unbound method `class.target_method` keep the same doc and signature
        get_bound_safe_patch_fn = update_wrapper_extended(get_bound_safe_patch_fn, original_fn.fget)
        safe_patch_obj = property(get_bound_safe_patch_fn)
    else:
        safe_patch_obj = update_wrapper_extended(safe_patch_function, original)

    new_patch = _wrap_patch(destination, function_name, safe_patch_obj)
    _store_patch(autologging_integration, new_patch)


def revert_patches(autologging_integration):
    """
    Reverts all patches on the specified destination class for autologging disablement
    purposes.

    :param autologging_integration: The name of the autologging integration associated with the
                                    patch. Note: If called via fluent api
                                    (`autologging_integration="mlflow"`), then revert all patches
                                    for all active autologging integrations.
    """
    for patch in _AUTOLOGGING_PATCHES.get(autologging_integration, []):
        gorilla.revert(patch)

    _AUTOLOGGING_PATCHES.pop(autologging_integration, None)


def update_wrapper_extended(wrapper, wrapped):
    """
    Update a `wrapper` function to look like the `wrapped` function. This is an extension of
    `functools.update_wrapper` that applies the docstring *and* signature of `wrapped` to
    `wrapper`, producing a new function.

    :return: A new function with the same implementation as `wrapper` and the same docstring
             & signature as `wrapped`.
    """
    updated_wrapper = functools.update_wrapper(wrapper, wrapped)
    # Assign the signature of the `wrapped` function to the updated wrapper function.
    # Certain frameworks may disallow signature inspection, causing `inspect.signature()` to throw.
    # One such example is the `tensorflow.estimator.Estimator.export_savedmodel()` function
    try:
        updated_wrapper.__signature__ = inspect.signature(wrapped)
    except Exception:
        print("Failed to restore original signature for wrapper around %s", wrapped)
    return updated_wrapper


def _wrap_patch(destination, name, patch_obj, settings=None):
    """
    Apply a patch.

    :param destination: Patch destination
    :param name: Name of the attribute at the destination
    :param patch_obj: Patch object, it should be a function or a property decorated function
                      to be assigned to the patch point {destination}.{name}
    :param settings: Settings for gorilla.Patch
    """
    if settings is None:
        settings = gorilla.Settings(allow_hit=True, store_hit=True)

    patch = gorilla.Patch(destination, name, patch_obj, settings=settings)
    gorilla.apply(patch)
    return patch


def _store_patch(autologging_integration, patch):
    """
    Stores a patch for a specified autologging_integration class. Later to be used for being able
    to revert the patch when disabling autologging.

    :param autologging_integration: The name of the autologging integration associated with the
                                    patch.
    :param patch: The patch to be stored.
    """
    if autologging_integration in _AUTOLOGGING_PATCHES:
        _AUTOLOGGING_PATCHES[autologging_integration].add(patch)
    else:
        _AUTOLOGGING_PATCHES[autologging_integration] = {patch}


__all__ = [
    "safe_patch",
    "PatchFunction",
    "update_wrapper_extended",
]
