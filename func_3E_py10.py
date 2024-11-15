# %%
# Functional Python Programming - Use a functional approach to write succinct, expressive, & efficient Python(2022, 3E, Pact).pdf
# p.4
# Comparing & contrasting proceduarl and functional styles:

# Procedural:
# The sum computed by this fx include only numbers that are multiples of 3 or 5.
# this fx is strictly procedural, avoiding explicit python object features.
# Functions state is defined by the values of the variables s and n.
# 	The variable n takes on values such that 1 <= n < 10.
# As the iteration involves an ordered exploration of values for the n variable,
# 	we can prove that itll terminate when the value of n is equal to the value of limit.
# There are two explicit assignment statements, both settings values for the s variable.
# 	These state changes are visible. The value of n is set implicitly by the fore statement.
#	The state change in the s variable is an essential element of the state of the computation.
def sum_numeric(limit: int = 10) -> int:
	s = 0
	for i in range(1, limit):
		if n % 3 == 0 or n % 5 == 0:
			s += n
	return s

# %%
# Look at this from a purely functional perspective. Then examine more pythonic perspective that retains the essence
# of a functional approach while leveraging a number of Pythons features.
# The sum of the multples of 3 and 5 can be decomposed into two parts:
# - The sum of a sequence of numbers.
# - A sequence of values that pass a simple test condition, for example, being multiples of 3 and 5.
# The sum of a sequence has a recursive definition.
from collections.abc import Sequence
def sumr(seq: Sequence[int]) -> int:
	if len(seq) == 0:
		return 0
	return seq[0] + sumr(seq[1:])

# %%
sumr([7, 11])

# %%
sumr([11])

# %%
sumr([])

# %%

"""
In this fx, compared a given value, v, against the upper bound, limit. If v has reached the upper bound, the resulting
list must be empty. This is the base case for the given recursion.
There are two more cases defined by an externally defined filter_func() fx. The value of v is passed to the filter_func() fx;
if this returns a very small list, containing one element, this can be concatenated with any remaining values computed by the until fx.
If the value of v is rejected by the filter_func() fx, the value is ignored and the result is simply defined by any remaining
values computed by the until() function.
"""
from collections.abc import Sequence, Callable

def until(limit: int, filter_func: Callable[[int], bool], v: int) ->list[int]:
	if v >= limit:
		return []
	elif filter_func(v):
		return [v] + until(limit, filter_func, v + 1)
	else:
		return until(limit, filter_func, v + 1)

"""
You can see that the value of v will increase from an initial val until it reaches limit, assuring
us that we'll reach the base case.
Before using the until() function, define a small fx to filter values that are multiples of 3 OR 5.
"""


def mult_3_5(x: int) -> bool:
	return x % 3 == 0 or x % 5 == 0

# couldve also been defined as a lambda object to emphasize succinct definitions of simple functions.

# The fx can be combined ith until() to generate a sequence of values which are multiples of 3 and 5

# %%
until(10, mult_3_5, 0)

# %%
# looking at the decomposition at the top (sumr) u can now compute sums and a way to compute the sequence of values.
# combining sumr() and until() to compute a sum of values:

def sum_functional(limit: int = 10) -> int:
	return sumr(until(limit, mult_3_5, 0))

# %%
# Its a purely functional, recursive definition that matches the mathematical abstractions. making it easier to reason about.
# functional hybrid

def sum_hybrid(limit: int = 10) -> int:
	return sum(
		n for n in range(1, limit)
		if n % 3 == 0 or n % 5 == 0
	)

# generator expression to iterate through a collection of values and compute the sum of these values.
# the range(1, 10) object is an iterable; it generates a sequence of values {n|1<= n < 10}.
# the more complex expression n for n in range(1, 10) if n % 3 == 0 or n % 5 == 0 is also a generator.
# it produces a set of values {n|1<= n < 10∧(𝑛 ≡ 0 mod 3∨𝑛 ≡ 0 mod 5)}
# can be described as "values of n such that n is less than or equal to n and n is less than 10 and n is equivalent to 0 modulo 3 or n is equivalent to 0 modulo 5"
# the variable n is bound, in turn, to each of the values provided by the range object. the sum fx consumes the iterable values, creating a final object, 23.
# * The bound variable, n, doesnt exist outside the generator expression. The variable n isnt visible elsewhere in the program.
# * a 'for' statement (outside the generator expression) creates a proper variable in the local namespace.
# * the generator expression doesnt create a variable in the same way that a 'for' statement does:

sum(n for n in range(1, 10) if n % 3 == 0 or n % 5 == 0)


# %%
# n
# returns NameError: name 'n' isnt defined.
# * The generator expression doesnt pollute the namespace with variables, like n, which arent relevant
# * outside the very narrow context of the expression.
# * Ch 1, pg 10.
# The functional programs in this book will rely on the following 3 stacks of abstractions:
# * Applications will be functions - all the way down - until you hit the objects.
# ** The underlying python runtime environment that supports the functional programming is objects - all the way down - until you hit the libraries.
# *** The libraries that suppport python are a turtle on which Python stands.

# %%
# * Newton-Raphson algorithm for locating any roots of a function.
# define function that will compute a square root of a number.
# backbone of this approximation is the calculation of the next approximation from the current approximation.
# the next_() function takes x, an approximation to the sqrt(n) value, and calculates the next value that brackets the proper root.
# ex:
def next_(n: float, x: float) -> float:
    return (x + n / x) / 2

# This fx computes a series of values that will quickly converge on some value x such that $x = n/x$, meaning x = sqrt(n)

# * Note: the name next() woulve collided with the builtin function 'next'. Calling it next_() lets you
# * follow the original presentation as closely as possible, using Pythonic names.
n = 2
f = lambda x: next_(n, x)
a0 = 1.0
[round(x, 4) for x in (a0, f(a0), f(f(a0)), f(f(f(a0))),)]

# %%
# Defined the function f() as a lambda that will converge on $sqrt(n)$ where $n = 2$. Starting at
# 1.0 as the initual value for a0. Then evaluated a sequence of recursive evaluations $a_{1} = f(a_{0})$, $a_{2} = f(a_{1})$, $a_{3} = f(a_{2})$, etc until the difference between successive values is less than 0.0001. The sequence converges on 1.41421.
# These functions evaluated these expressions using a generator expression so that you could round each value to 4 decimal places. This makes the output easier to read and easier to use with 'doctest'. The sequence appears to converge rapidly to $sqrt(2)$. To get a more precise answer, you must continue to perform the series of steps after the first four.
# This is a function that will (in principle) generate an infinite sequence of $a_{i}$ values. This series will converge
# on the proper square root:
#

# %%
from collections.abc import Iterator, Callable

def repeat(f: Callable[[float], float], a: float) -> Iterator[float]:
	yield a
	yield from repeat(f, f(a))

"""
This function will generate a sequence of approximations using a function $f()$ and an initial approximation $a$.
"""

# %%
# in python you need to be a little clever when taking items from an inifinte sequence one at a time.
# it works out well to use a simple interface function that wraps a slightly more complex recursion.

from collections.abc import Iterator
def within(𝜖: float, iterable: Iterator[float]) -> float:
	def head_tail(𝜖: float, a: float, iterable: Iterator[float]) -> float:
		b = next(iterable)
		if abs(a - b) <= 𝜖:
			return b
		return head_tail(𝜖, b, iterable)
	return head_tail(𝜖, next(iterable), iterable)


# %%
#
