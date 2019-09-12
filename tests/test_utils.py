import unittest

from puzzlesolver.utils import halver

class TestUtilities(unittest.TestCase):
    def test_can_divide_by_two(self):
        my_halver = halver()
        self.assertEqual(next(my_halver), 0.5)
