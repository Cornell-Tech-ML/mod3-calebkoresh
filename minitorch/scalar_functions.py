from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the input values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for the Add function.

        Args:
        ----
            ctx: Context object to save values.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: Result of the Add function.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for the Add function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            Tuple[float, ...]: Derivatives of the input values with respect to the output.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the Log function.

        Args:
        ----
            ctx: Context object to save values.
            a: Input value.

        Returns:
        -------
            float: Result of the Log function.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the Log function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            float: Derivative of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for the Mul function.

        Args:
        ----
            ctx: Context object to save values.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: Result of the Mul function.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for the Mul function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            Tuple[float, float]: Derivatives of the input values with respect to the output.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the Inv function.

        Args:
        ----
            ctx: Context object to save values.
            a: Input value.

        Returns:
        -------
            float: Result of the Inv function.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the Inv function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            float: Derivative of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the Neg function.

        Args:
        ----
            ctx: Context object to save values.
            a: Input value.

        Returns:
        -------
            float: Result of the Neg function.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the Neg function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            float: Derivative of the input with respect to the output.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the Sigmoid function.

        Args:
        ----
            ctx: Context object to save values.
            a: Input value.

        Returns:
        -------
            float: Result of the Sigmoid function.

        """
        ctx.save_for_backward(operators.sigmoid(a))
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the Sigmoid function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            float: Derivative of the input with respect to the output.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the ReLU function.

        Args:
        ----
            ctx: Context object to save values.
            a: Input value.

        Returns:
        -------
            float: Result of the ReLU function.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the ReLU function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            float: Derivative of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for the Exp function.

        Args:
        ----
            ctx: Context object to save values.
            a: Input value.

        Returns:
        -------
            float: Result of the Exp function.

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for the Exp function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            float: Derivative of the input with respect to the output.

        """
        out: float = ctx.saved_values[0]
        return out * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1$ if $x < y$ else $0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for the LT function.

        Args:
        ----
            ctx: Context object to save values.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: Result of the LT function.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for the LT function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            Tuple[float, float]: Derivatives of the input values with respect to the output.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = 1$ if $x == y$ else $0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for the EQ function.

        Args:
        ----
            ctx: Context object to save values.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: Result of the EQ function.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for the EQ function.

        Args:
        ----
            ctx: Context object containing saved values.
            d_output: Derivative of the output with respect to the function's output.

        Returns:
        -------
            Tuple[float, float]: Derivatives of the input values with respect to the output.

        """
        return 0.0, 0.0
