---
title: functional programming 2E py312
format:
  html:
    code-fold: true
jupyter: WinPy3.12sys
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: WinPy3.12
    language: python
    name: win_pyenv3.12
---

```python
#| label: fig-polar
#| fig-cap: "A line plot on a polar axis"

import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r
fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
ax.plot(theta, r)
ax.set_rticks([0.5, 1, 1.5, 2])
ax.grid(True)
plt.show()
```


Functional Python Programming - Use a functional approach to write succinct, expressive, & efficient Python(2022, 3E, Pact).pdf pg.4
Comparing & contrasting proceduarl and functional styles:

- Procedural:
The sum computed by this fx include only numbers that are multiples of 3 or 5.
this fx is strictly procedural, avoiding explicit python object features.
Functions state is defined by the values of the variables s and n.
The variable n takes on values such that 1 <= n < 10.
As the iteration involves an ordered exploration of values for the n variable,
we can prove that itll terminate when the value of n is equal to the value of limit.
There are two explicit assignment statements, both settings values for the s variable.
These state changes are visible. The value of n is set implicitly by the fore statement.
The state change in the s variable is an essential element of the state of the computation.


```python
def sum_numeric(limit: int = 10) -> int:
    s = 0
    for i in range(1, limit):
        if n % 3 == 0 or n % 5 == 0:
            s += n
    return s
```

Look at this from a purely functional perspective. Then examine more pythonic perspective that retains the essence
of a functional approach while leveraging a number of Pythons features.
The sum of the multples of 3 and 5 can be decomposed into two parts:

- The sum of a sequence of numbers.
- A sequence of values that pass a simple test condition, for example, being multiples of 3 and 5.
  The sum of a sequence has a recursive definition.


```python
from collections.abc import Sequence


def sumr(seq: Sequence[int]) -> int:
    if len(seq) == 0:
        return 0
    return seq[0] + sumr(seq[1:])
```

```python
sumr([7, 11])
```

```python
sumr([11])
```

```python
sumr([])
```

In this fx, compared a given value, v, against the upper bound, limit. If v has reached the upper bound, the resulting
list must be empty. This is the base case for the given recursion.
There are two more cases defined by an externally defined filter_func() fx. The value of v is passed to the filter_func() fx;
if this returns a very small list, containing one element, this can be concatenated with any remaining values computed by the until fx.
If the value of v is rejected by the filter_func() fx, the value is ignored and the result is simply defined by any remaining
values computed by the until() function.


```python
from collections.abc import Sequence, Callable


def until(limit: int, filter_func: Callable[[int], bool], v: int) -> list[int]:
    if v >= limit:
        return []
    elif filter_func(v):
        return [v] + until(limit, filter_func, v + 1)
    else:
        return until(limit, filter_func, v + 1)
```

You can see that the value of v will increase from an initial val until it reaches limit, assuring
us that we'll reach the base case.
Before using the until() function, define a small fx to filter values that are multiples of 3 OR 5.


```python
def mult_3_5(x: int) -> bool:
    return x % 3 == 0 or x % 5 == 0
```

couldve also been defined as a lambda object to emphasize succinct definitions of simple functions.

The fx can be combined ith until() to generate a sequence of values which are multiples of 3 and 5


```python
until(10, mult_3_5, 0)
```

looking at the decomposition at the top (sumr) u can now compute sums and a way to compute the sequence of values.
combining sumr() and until() to compute a sum of values:


```python
def sum_functional(limit: int = 10) -> int:
    return sumr(until(limit, mult_3_5, 0))
```

Its a purely functional, recursive definition that matches the mathematical abstractions. making it easier to reason about.
functional hybrid


```python
def sum_hybrid(limit: int = 10) -> int:
    return sum(n for n in range(1, limit) if n % 3 == 0 or n % 5 == 0)
```

Generator expression to iterate through a collection of values and compute the sum of these values.
the range(1, 10) object is an iterable; it generates a sequence of values ${n|1<= n < 10}$.
the more complex expression `n for n in range(1, 10) if n % 3 == 0` or `n % 5 == 0` is also a generator.
it produces a set of values ${n|1<= n < 10 \And ( n is 0 \mod 3{∨} || n ≡ 0 \mod 5)}$.
can be described as "values of n such that n is less than or equal to n and n is less than 10 and n is equivalent to 0 modulo 3 or n is equivalent to 0 modulo 5"
the variable n is bound, in turn, to each of the values provided by the range object. the sum fx consumes the iterable values, creating a final object, 23.

- The bound variable, n, doesnt exist outside the generator expression. The variable n isnt visible elsewhere in the program.
- A 'for' statement (outside the generator expression) creates a proper variable in the local namespace.
- The generator expression doesnt create a variable in the same way that a 'for' statement does:


```python
sum(n for n in range(1, 10) if n % 3 == 0 or n % 5 == 0)
```


- The generator expression doesnt pollute the namespace with variables, like n, which arent relevant
- outside the very narrow context of the expression.
- Ch 1, pg 10.
  The functional programs in this book will rely on the following 3 stacks of abstractions:
- Applications will be functions - all the way down - until you hit the objects.

  - The underlying python runtime environment that supports the functional programming is objects - all the way down - until you hit the libraries.
  - The libraries that suppport python are a turtle on which Python stands.

- Newton-Raphson algorithm for locating any roots of a function.
  define function that will compute a square root of a number.
  backbone of this approximation is the calculation of the next approximation from the current approximation.
  the next\_() function takes x, an approximation to the sqrt(n) value, and calculates the next value that brackets the proper root.
  ex:


```python
def next_(n: float, x: float) -> float:
    return (x + n / x) / 2
```

This fx computes a series of values that will quickly converge on some value x such that $x = n/x$, meaning x = sqrt(n)

- Note: the name next() woulve collided with the builtin function 'next'. Calling it next\_() lets you
- follow the original presentation as closely as possible, using Pythonic names.

### page 12


```python
n = 2
f = lambda x: next_(n, x)
a0 = 1.0
[round(x, 6) for x in (a0, f(a0), f(f(a0)), f(f(f(a0))), f(f(f(f(a0)))))]
```

Defined the function f() as a lambda that will converge on $sqrt(n)$ where $n = 2$. Starting at
1.0 as the initual value for a0. Then evaluated a sequence of recursive evaluations $a_{1} = f(a_{0})$, $a_{2} = f(a_{1})$, $a_{3} = f(a_{2})$, etc until the difference between successive values is less than 0.0001. The sequence converges on 1.41421.
These functions evaluated these expressions using a generator expression so that you could round each value to 4 decimal places. This makes the output easier to read and easier to use with 'doctest'. The sequence appears to converge rapidly to $sqrt(2)$. To get a more precise answer, you must continue to perform the series of steps after the first four.
This is a function that will (in principle) generate an infinite sequence of $a_{i}$ values. This series will converge
on the proper square root:


```python
from collections.abc import Iterator, Callable


def repeat(f: Callable[[float], float], a: float) -> Iterator[float]:
    yield a
    yield from repeat(f, f(a))
```

This function will generate a sequence of approximations using a function $f()$ and an initial approximation $a$.

in python you need to be a little clever when taking items from an inifinte sequence one at a time.
it works out well to use a simple interface function that wraps a slightly more complex recursion.


```python
from collections.abc import Iterator


def within(𝜖: float, iterable: Iterator[float]) -> float:
    def head_tail(𝜖: float, a: float, iterable: Iterator[float]) -> float:
        b = next(iterable)
        if abs(a - b) <= 𝜖:
            return b
        return head_tail(𝜖, b, iterable)

    return head_tail(𝜖, next(iterable), iterable)
```

Using the three functions, next\_(), repeat(), and within(), to create a square root function as follows:


```python
def sqrt(n: float) -> float:
    return within(𝜖=0.0001, iterable=repeat(lambda x: next_(n, x), 1.0))
```

A more advanced version could use default param values to make changes possible. Exercise:
the definition of `sqrt()` should be rewritten so an expression such as `sqrt(1.0, 0.000_01, 3)` will start with an approximation of 1.0 and compute the value of $\sqrt{3}$ to within 0.000*01. For most applications, the initial a*{0} value can be 1.0. The closer it is to the actual square root, the more rapidly this algorithm converges.

# Note:

The within function done above was written to match the original articles definition. Pythons `itertools` lib provides a `takewhile()` function that might be better for this application than the `within()` function. Similarly the `math.isclose()` func may be better than $abs(a-b) <= \epsilon$ used here. Python offers many pre-built functional programming features, check in Chapter 8 (itertools).

## Histogram

The objective of the exercise is to compute a histogram of the most popular ZIP codes in
the source data file. The data must be cleansed to have the following two ZIP formats:
• Five characters with no post-office box, for example 03801
• Ten characters with a hyphen, for example 03899-9876

(pg.21)


```python
from collections import Counter
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

DEFAULT_PATH = Path.cwd() / "address.csv"


def main(source_path: Path = DEFAULT_PATH) -> None:
    freq: Counter[str] = Counter()
    with source_path.open() as source:
        rdr = csv.DictReader(source)
        for row in rdr:
            if "-" in row["ZIP"]:
                text_zip = row["ZIP"]
                missing_zeros = 10 - len(text_zip)
                if missing_zeros:
                    text_zip = missing_zeros * "0" + text_zip
            else:
                text_zip = row["ZIP"]
                if 5 < len(row["ZIP"]) < 9:
                    missing_zeros = 9 - len(text_zip)
                else:
                    missing_zeros = 5 - len(text_zip)
                if missing_zeros:
                    text_zip = missing_zeros * "0" + text_zip
            freq[text_zip] += 1
    print(freq)


if __name__ == "__main__":
    main()

if TYPE_CHECKING:

    def zip_histogram(reader: csv.DictReader[str]) -> Counter[str]:
        pass


import pytest


@pytest.fixture
def mock_file(tmp_path: Path) -> Path:
    data_path = tmp_path / "data.csv"
    rows: list[dict[str, Any]] = [
        {"ZIP": 3801, "CITY": "Portsmouth", "STATE": "NH"},
        {"ZIP": 12345, "CITY": "Schenectady", "STATE": "NY"},
        {"ZIP": 641, "CITY": "Arecibo", "STATE": "PR"},
        {"ZIP": 38011234, "CITY": "Portsmouth", "STATE": "NH"},
        {"ZIP": "3801-1234", "CITY": "Portsmouth", "STATE": "NH"},
        {"ZIP": "12345-2356", "CITY": "Schenectady", "STATE": "NY"},
        {"ZIP": "641-1234", "CITY": "Arecibo", "STATE": "PR"},
    ]
    with data_path.open("w", newline="") as data_file:
        wtr = csv.DictWriter(data_file, ["ZIP", "CITY", "STATE"])
        wtr.writeheader()
        wtr.writerows(rows)
    return data_path


def test_zip_histogram(mock_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(mock_file)
    out, err = capsys.readouterr()
    assert out.splitlines() == [
        "Counter({'03801': 1, '12345': 1, '00641': 1, '38011234': 1, '03801-1234': 1, '12345-2345': 1, '00641-1234': 1})"
    ]

```

It can be refactored (tangled also so that quarto works with it) into two parts.
new `zip_histogram()` should be written to contain much of the processing detail. The func will process the opened file, returning a Counter object.


```python
def zip_histogram(reader: csv.DictReader[str]) -> Counter[str]:
    return Counter()

```

1. The `main()` func is left with the responsibility to open the file, create the csv.DictReader instance, eval `zip_histogram()`, and print the histogram.
2. Once the zip_histogram function is defined, the cleansing of the ZIP attribute can be refactored into separate functions, with a name like `zip_cleanse()`. Instead of setting the value of the `text_zip` variable, this function can return a cleansed result. This can then be tested separately to be sure the various cases are handled gracefully.
3. The distinction between long ZIP codes with a hyphen and without is something that should be fixed. Once 'cleansed' works, note: add another function to inject hyphens into ZIP codes with only digits. ex: transforming 39011345 to 3901-1345.
   The final zip function should look like:


```python
def zip_histogram(reader: csv.DictReader[str]) -> Counter[str]:
    return Counter(zip_cleanse(row["ZIP"]) for row in reader)
```

# Chapter 2: Essential Functional Concepts

## Functions as First-class objects && Immutable data


```python
def example(a, b, **kwargs):
    return a * b


print(type(example))
print(example.__code__.co_varnames)
print(example.__code__.co_argcount)

```

- The `__code__` _attribute_ of the function object has attributes of its own. Functions can be manipulated like all other objects.
- Pure functions are easier to test and are conceptually simpler.
  - Write local-only code. Meaning avoid all _global_ statements.


```python
mersenne = lambda x: 2**x - 1
mersenne(17)
# 131071

default_zip = lambda row: row.setdefault("ZIP", "00000")

r_0 = {"CITY": "Vaca Key"}
print(default_zip(r_0))
r_0  # {'CITY': 'Vaca Key', 'ZIP': '00000'}

r_1 = {"CITY": "Sterling Heights", "ZIP": "48310"}
print(default_zip(r_1))  # 49310

```

### Higher-order Functions

Functions that accept a function as an _argument_ or return a function as a value. Leading to composite functions.


```python md-indent=" "
year_cheese = [
    (200, 29.87),
    (2001, 30.12),
    (2002, 30.6),
    (2003, 30.66),
    (2004, 31.33),
    (2005, 32.62),
    (2006, 32.73),
    (2007, 33.5),
    (2008, 32.84),
    (2009, 33.02),
    (2010, 32.92),
]


max(year_cheese)


max(year_cheese, key=lambda yc: yc[1])  # (2007, 33.5)

```

In this example `max()` applies a lambda and returns the tuple with the largest value in position one of each tuple.

- Chapter 5 has more information on higher-order functions.



This snippet is from Chapter 11 of Fluent Python.

```python
from fluent.Vector2d import Vector2d
```

```python
v1 = Vector2d(3, 4)
print(v1.x, v1.y)
```

```python
x, y = v1
x, y
```

```python
v1
```

```python
v1_clone = eval(repr(v1))
v1 == v1_clone
```

```python
print(v1)
```

```python
octets = bytes(v1)
octets
```

```python
abs(v1)
```

```python
bool(v1), bool(Vector2d(0, 0))
```

## Classmethod vs. staticmethod
The `classmethod` decorator isnt mentioned in many Python tutorials, and netierh is staticmethod. Anyone who has learned OO in Java may wonder why Python has both of these decorators and not just one of them.
In the example above: to define a method that operates on a class and not on instances. `classmethod` changes the way the method is called, so it receives the class itself as the first argument, instead of an instance. Its most common use is for alternative constructors, like `frombytes`. Note how the last lineof `frombytes` actually uses the `cls` argument by invoking it to build a new instance: `cls(*memv)`.

In contrast, the `staticmethod` decorator changes a method so that it receives no special first argument. In essence, a static method is just a plain function that happens to live in a class body, instead of being defined at the module level. Example below contrasts the operation of `classmethod` and `staticmethod`:

```python
class Demo:
    @classmethod
    def klassmeth(*args):
        return args  #1
    @staticmethod
    def statmeth(*args):
        return args  # 2
```

```python
Demo.klassmeth() #3
```

```python
Demo.klassmeth('spam')
```

```python
Demo.statmeth() #4
```

```python
Demo.statmeth('spam')
```

### Fluent Python Ch 11. Pg. 365-370
1. klassmeth just returns all positional arguments.
2. statmeth does the same.
3. No matter how you invoke it, `Demo.klassmeth` receives the `Demo` class as the first argument.
4. `Demo.statmeth` behaves just like a plain old function.

The `classmethod` decorator is clearly useful, but good use cases for `staticmethod` are very rare (in the authors experience). Maybe the function is closely related even if it never touches the class, so you may want to place it nearby in the code. Even then, defining the function right before or after the class in the same module is close enough most of the time.
