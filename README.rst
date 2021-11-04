Autotest: Simpler Test Runner for Python
========================================

Goals
-----

We want a simpler test runner. One that:

#) works in a familiar way, 
#) does not impose restrictions or enforce a way of coding,
#) components do one thing each, and orthogonal to each other,
#) fits into existing libraries, notably Python itself,
#) keeps programmers close to their code,
#) makes (large) refactorings easy,
#) fits into a fast REPL-style way of working,
#) seamlessly scales from microtests to systemtests,
#) supports (continuous) integration,
#) puts the programmer in control.

That is a lot of requirements, but luckily this will fullfil all of them:

.. code:: python

    def some_function():
        some_more()


As long as out test runner is close to a simple Python function, we might achieve our goals.

So, we  make a library of simple combinable functions.


The Simplest Thing That Could Possibly Work
-------------------------------------------

1) Python's built-in `assert` is our starting point:

.. code:: python
   
   assert 1 == x, "x must be 1"
   

This is more profound than it seems.  Python's `assert` performs a test (simple truth value check) and immediately reports failure (by raising an AssertionError). In particular:
   
- there are no gathering, execution and reporting phases,
- the first failure stops the program, and
- discovering tests follows program execution

It can't be stressed enough how much these three points contribute to the goals above. Take a moment to reflect on this.

More on discovery later.


2) ``assert`` raises AssertionError without info. For expressive asserts and improved reporting autotest has (read 'test' as a verb):

.. code:: python

   test.eq(1, x)
   
This fetches ``eq`` from the built-in module ``operator``,  applies it and raises AssertionError including the operation and its arguments. It works stand-alone, there is no requirement for a certain context.


3) The second extension is a way to have simple fixtures (which are just context managers):

.. code:: python

   with test.tmp_path as p:
      assert p.exists()

   with test.raises(KeyError):
      {}[1]
   
These work in their own right. You could also use a context manager from contextlib. It is easy to create fixtures like the one used above:

.. code:: python
   @test.fixture
   def raises(exception):
     try:
       yield
     except exception:
       pass
     else:
       raise AssertionError(f"should raise {exception.__name}")
   

The fixtures are more versatile context managers as we see later.

 
4) Functions provide more context to tests/asserts. A function explicitly marked with ``@test`` groups tests, report their succes as a whole and accepts options:

.. code:: python

      @test
      def any_function(option=Value):
          assert 1 == 2
    

Read ``@test`` as a verb: the function is excuted immediately. Options at time of writing are:

======  =======  =======   ==========================================================
option  type     default   Explanation
======  =======  =======   ==========================================================
keep    boolean  False     Keep the function after running instead of discading it.
skip    boolean  False     Skip running, False for parent, True for child processes.
report  boolean  True      Report the succes or be silent.
======  =======  =======   ==========================================================


5) Test functions can declare fixtures by specifying them as arguments, like is done in pyttest:

.. code:: python

    @test
    def temp_file_usage(tmp_path):
        path = tmp_path / 'ape'
        path.write_text("hello")


Sorry, *WORK IN PROGRESS* ahead


..
    AssertionError
    826  	@test
    827  	def msg():
    828  	    x = 2
    829  ->	    assert 1 == x
    > /home/seecr/development/autotest/autotest.py(829)msg()
    -> assert 1 == x
    (Pdb) p x
    2
    (Pdb)


#) Pythons ``import`` is enough to find all tests. This mechanism is well-known, transparent, hierarchical and order preserving. Importing a module causes all tests to be run as a prerequisite.

Runing all tests is simply done by importing:

.. code:: sh

      $ python <module>

or when you module has ``__main__``:

.. code:: sh

      $ python -c "import <module>"
  
or when you want to test a submodule and its dependencies, just import that submodule.


