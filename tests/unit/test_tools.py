"""Regression tests for guidance._tools.Tool.call error handling.

See https://github.com/guidance-ai/guidance/issues/1485: ``Tool.call`` used to
catch ``BaseException`` (swallowing ``KeyboardInterrupt``/``SystemExit`` and
turning them into tool-output strings) and asserted that the traceback always
had an inner frame (crashing with a bare ``AssertionError`` when the callable
was not actually callable).
"""

import pytest
from guidance._tools import Tool


def _tool(fn):
    return Tool.from_callable(fn, name="t", description="d")


def test_call_propagates_keyboard_interrupt():
    def raises_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _tool(raises_keyboard_interrupt).call()


def test_call_propagates_system_exit():
    def raises_system_exit():
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        _tool(raises_system_exit).call()


def test_call_formats_regular_exception():
    def raises_value_error():
        raise ValueError("kaboom")

    result = _tool(raises_value_error).call()
    assert isinstance(result, str)
    assert "ValueError: kaboom" in result
    # The traceback should start inside the callable, not inside Tool.call.
    assert "raises_value_error" in result


def test_call_non_callable_reports_type_error():
    class NotCallable:
        pass

    tool = _tool(lambda: None)
    # Simulate a non-callable value assigned to .callable: the TypeError is then
    # raised at the call expression itself, so there is no inner traceback frame.
    tool.callable = NotCallable()
    result = tool.call()
    assert isinstance(result, str)
    assert "not callable" in result
