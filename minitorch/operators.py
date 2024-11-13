"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul


def mul(x: float, y: float) -> float:
    """Multiplies two numbers

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The product of x and y

    """
    return x * y


# - id


def id(id: float) -> float:
    """Return the input unchanged

    Args:
    ----
        id (float): Input number

    Returns:
    -------
        float: id

    """
    return id


# - add


def add(x: float, y: float) -> float:
    """Adds two numbers

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The sum of x and y

    """
    return x + y


def sub(x: float, y: float) -> float:
    """Subtracts two numbers

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The difference of x and y

    """
    return x - y


# - neg


def neg(x: float) -> float:
    """Negates a number

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The negation of x

    """
    return -x


# - lt


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        bool: True if x is less than y, False otherwise

    """
    return x < y


# - eq


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        bool: True if x is equal to y, False otherwise

    """
    return x == y


# - max


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The maximum of x and y

    """
    return x if x > y else y


# - is_close


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        bool: True if x and y are close, False otherwise

    """
    return abs(x - y) < 1e-2


# - sigmoid


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The sigmoid of x

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


# - relu


def relu(x: float) -> float:
    """Applies the ReLU activation function

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The ReLU of x

    """
    return float(x) if x >= 0 else float(0)


# - log


def log(x: float) -> float:
    """Calculates the natural logarithm

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The natural log of x

    """
    return math.log(x)


# - exp


def exp(x: float) -> float:
    """Calculates the exponential function

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The exponential of x

    """
    return math.exp(x)


# - log_back


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg

    Args:
    ----
        x (float): Input number
        d (float): Derivative of the output with respect to x

    Returns:
    -------
        float: The derivative of the log function with respect to x

    """
    return d / x


# - inv


def inv(x: float) -> float:
    """Calculates the reciprocal

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The inverse of x

    """
    return 1 / x


# - inv_back


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg

    Args:
    ----
        x (float): Input number
        d (float): Derivative of the output with respect to x

    Returns:
    -------
        float: The derivative of the inverse function with respect to x

    """
    return -d / (x * x)


# - relu_back


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg

    Args:
    ----
        x (float): Input number
        d (float): Derivative of the output with respect to x

    Returns:
    -------
        float: The derivative of the ReLU function with respect to x

    """
    return d * (x > 0)


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map


def map(fn: Callable[[float], float], iter: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable

    Args:
    ----
        fn (Callable[[float], float]): Function to apply to each element
        iter (Iterable[float]): Iterable to apply the function to

    Returns:
    -------
        Iterable[float]: Iterable with the function applied to each element

    """
    return [fn(x) for x in iter]


# - zipWith


def zipWith(
    fn: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that applies a given function to two iterables

    Args:
    ----
        fn (Callable[[float, float], float]): Function to apply to each pair of elements
        iter1 (Iterable[float]): First iterable
        iter2 (Iterable[float]): Second iterable

    Returns:
    -------
        Iterable[float]: Iterable with the function applied to each pair of elements

    """
    return [fn(x, y) for x, y in zip(iter1, iter2)]


# - reduce


def reduce(
    fn: Callable[[float, float], float], iter: Iterable[float], init: float
) -> float:
    """Higher-order function that reduces an iterable to a single value

    Args:
    ----
        fn (Callable[[float, float], float]): Function to apply to the iterable
        iter (Iterable[float]): Iterable to reduce
        init (float): Initial value for the reduction

    Returns:
    -------
        float: Reduced value

    """
    result = init
    for x in iter:
        result = fn(result, x)
    return result


#
# Use these to implement
# - negList : negate a list


def negList(iter: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map

    Args:
    ----
        iter (Iterable[float]): List to negate

    Returns:
    -------
        Iterable[float]: Negated list

    """
    return map(neg, iter)


# - addLists : add two lists together


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith

    Args:
    ----
        iter1 (Iterable[float]): First list
        iter2 (Iterable[float]): Second list

    Returns:
    -------
        Iterable[float]: Sum of the two lists

    """
    return zipWith(add, iter1, iter2)


# - sum: sum lists


def sum(iter: Iterable[float]) -> float:
    """Sum all elements in a list using reduce

    Args:
    ----
        iter (Iterable[float]): List to sum

    Returns:
    -------
        float: Sum of the list

    """
    return reduce(add, iter, 0.0)


# - prod: take the product of lists


def prod(iter: Iterable[float]) -> float:
    """Multiply all elements in a list using reduce

    Args:
    ----
        iter (Iterable[float]): List to multiply

    Returns:
    -------
        float: Product of the list

    """
    return reduce(mul, iter, 1.0)


# TODO: Implement for Task 0.3.
