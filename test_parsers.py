test_parsers.py

import unittest
import parsers

class ParsersTest(unittest.TestCase):

    #def test_partition(self):
    #    x = [1, 2, 2, 1]
    #    y = [5, 6, 7, 8]
    #    (result, errors) = fieldparsers.partition_values(x, y, [1, 2]);
    #    self.assertEqual({1: [5, 8], 2:[6, 7]}, result)

    answer_to_booked_loans = 
    def test_booked_loans(self):
    	for each in self:
    		self.asertEquals('Booked Loan', parsers.only_booked_loans(each))

    def test_parse_employment_years(self):
        self.assertEquals(1, parsers.parse_employment_years('1 year'))
        self.assertEquals(0, parsers.parse_employment_years('<1 year'))
        self.assertEquals(10, parsers.parse_employment_years('10+ years'))

if __name__ == '__main__':
    unittest.main()