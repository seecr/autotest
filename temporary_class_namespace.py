

# This works:

def f():

    def g():
        print("g(1)")

    g()

f()


def f():

    def g():
        print("g(2)")

    def h():
        g()

    h()

f()


# This also works:

class f():

    def g():
        print("g(3)")

    g()


# but this does not:

class f():

    def g():
        print("g(4)")

    def h():
        g() # "g is not defined"

    h()


# Explanation:
#
# 1.
# During the definition of def f, nothing is executed.
# During the definition of class f, the class body is executed.
# This is why f() is needed on line 12 and 25.
#
# 2.
# The def f() is completely defined before being executed: it's local namespace includes g.
# The class f() is incomplete when it is executed, g is only in a temporary namespace.
# Function h has its own namespace and has no access to class f's temporary namespace.
#
# 3.
# Calling g() directly does work, as this happens in the same temporary namespace.
