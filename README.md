Autotest: Test Runner for Python
================================

The Simplest Thing That Could Possibly Work
-------------------------------------------

Building large, robust, world class applications toughts us that we need simple, straightforward little tools, such as:

1. terminal, shell, coreutils etc
2. vi, with as few plugins as possible
3. python

These tools work in a familiar way, and do one thing each. They keep programmers close to their code.

For testing, we want a tool like that.


Testing Vision
--------------

1. Tests must be first class citizens: just code, and an inherent part of the application. Tests can do anything and are not limited in scope. A test could assert the presence of certain hardware and prope expected behavior. A test could also verify the proper startup of an application, given all dependencies, initialization etc. A test could do anything in between.

2. Testing is not confined to a separate stage during development but may happen at any point in time, particularly also at runtime, for example during startup. Think of CPU's doing a self-test. Tests could also run while the application is running, for example when a new module is added dynamically.

3. Testing may influence the way an application is structured and build (test-driven development already does this to a large extend) in order to facilitate testing. Testing support can be present in any part of the application, by means of test arguments, test functions, test classes, test modes, whatever is deemed necessary.


The Simplest Solution
-----------------

1. Pythons `import` is enough to find all tests. This mechanism is well-known, transparent, hierarchical and order preserving. Importing a module causes all tests to be run as a prerequisite.

  Runing all tests is simply done by importing:

  ```bash
      $ python <module>
  ```
  or when you module has `__main__`:

  ```bash
      $ python -c "import <module>"
  ```
  or when you want to test a submodule and its dependencies, just import that submodule.

2. Tests are normal functions, explicitly marked using a language feature: a decorator.
  ```python
      @test
      def any_function():
          assert 1 == 2
  ```
  The tests of autotest itself contain nice examples of what is possible using grouping in classes, bootstrapping, testing tests etc.

3. Tests are executed (and reported) synchronously with program execution and stops at first failure. This gives focus and allows for starting a debugger at the right spot in your code, with all the context you need. It also allows for testing while parts of the application do not even compile yet (e.g. during refactoring).

4. Simple fixtures are possible by specifying them as arguments, like is done in pyttest.
  ```python
    @fixture
    def tmp_path():
        p = pathlib.Path(tempfile.mkdtemp())
        yield p
        shutil.rmtree(p)

    @test
    def temp_file_usage(tmp_path):
        path = tmp_path / 'ape'
        path.write_text("hello")

5. assert and test
  Any AssertionError is dealt with, but using Python\'s `assert` does not give much information. It is recommended to raise AssertionErrors with at least arguments. A helpers exists to use `operator` and raise exceptions with the args set to the args of the operator.
```python
@test
def use_test_operator():
    test.eq(1, 1)
    test.lt(1, 2) # etc
```
The `test` decorator doubles as a tool to invoke and assert any operator from Python `operator`.
