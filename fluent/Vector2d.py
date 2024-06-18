import math as mt  # noqa: INP001
from array import array

# class Vector:

#     def __init__(self, x = 0, y = 0):
#         self.x = x
#         self.y = y

#     def __repr__(self):
#         return f'Vector({self.x!r}, {self.y!r})'

#     def __abs__(self):
#         return mt.hypot(self.x, self.y)

#     def __bool__(self):
#         return bool(abs(self))

#     def __add__(self, other):
#         x = self.x + other.x
#         y = self.y + other.y
#         return Vector(x, y)

#     def __mul__(self, scalar):
#         return Vector(self.x * scalar, self.y * scalar)


class Vector2d:
    """
    1. Typecode is a class attribute that is used to convert Vector2d
        instances to/from bytes
    2. Converting x & y to float in __init__ catches errors early, which is
        helpful in case Vector2d is called with unsuitable args
    3. __iter__ makes a Vector2d iterable; this is what makes unpacking work
        (ex: x, y = my_vector). We implement it simply by using a generator
        expression to yield the components one after the other.
    4. __repr__ builds a string by interpolating the components with {!r} to get
        their repr; because Vector2d is iterable, *self feeds the x & y components
        to format.
    5. From an iterable Vector2d, it's easy to build a tuple for display as an
        ordered pair.
    6. To generate bytes, we convert the typecode to bytes & concat.
    7. .. bytes converted from array built by iterating over the instance.
    8. To quickly compare all components, build tuples out of operands.
        This works for operands that are instances of Vector2d, but has issues.
        See warning.
    9. The magnitude is the length of the hypotenuse of the right triangle formed
        by the x & y components.
    10. __bool__ uses abs(self) to compute the magnitude, then converts it to bool,
        so 0.0 becomes False, nonzero is True.

    """

    typecode = "d"  # 1

    def __init__(self, x, y):
        self.x = float(x)  # 2
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))  # 3

    def __repr__(self):
        class_name = type(self).__name__
        return "{}({!r}, {!r})".format(class_name, *self)  # 4

    def __str__(self):
        return str(tuple(self))  # 5

    def __bytes__(self):
        return bytes([ord(self.typecode)]) + bytes(array(self.typecode, self)) # 6, 7

    def __eq__(self, other):
        return tuple(self) == tuple(other) # 8

    def __abs__(self):
        return mt.hypot(self.x, self.y) # 9

    def __bool__(self):
        return bool(abs(self)) # 10

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

"""
# Fluent Python, Ch 11. pg. 365-370
1. the class method decorator modifies a method so it can be called directly on a class.
2. No self arg; instead, the class itself is passed as the first arg - conventionally named cls.
3. Read the typecode from the first byte.
4. Create a memoryview from the octets binary sequence and use the typecode to cast it.
5. Unpack the memoryview resulting from the cast into the pair of args needed for the constructor.
"""

@classmethod  # 1
def frombytes(cls, octets):  # 2
    typecode = chr(octets[0])  # 3
    memv = memoryview(octets[1:]).cast(typecode)  # 4
    return cls(*memv) # 5
