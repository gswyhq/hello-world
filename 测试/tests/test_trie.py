#!/usr/bin/python3
# coding: utf-8

import unittest
from pytrie import SortedStringTrie


class TestTrie(unittest.TestCase):

    def setUp(self):
        """
        测试前准备环境的搭建(setUp)
        :return: 
        """
        self.words = 'an ant all allot alloy aloe are ate be'.split()
        self.trie = SortedStringTrie(zip(self.words, range(len(self.words))))

    def test_longest_prefix(self):
        """
        定义了以'test'开头的方法, 即是一个测试用例
        :return: 
        """
        self.assertEqual(self.trie.longest_prefix('antonym'), 'ant')
        self.assertEqual(self.trie.longest_prefix('are'), 'are')
        self.assertEqual(self.trie.longest_prefix('alla'), 'all')
        self.assertEqual(self.trie.longest_prefix('allo'), 'all')
        self.assertRaises(KeyError, self.trie.longest_prefix_item, 'alumni')
        self.assertEqual(self.trie.longest_prefix('alumni', default=None), None)
        self.assertEqual(self.trie.longest_prefix('linux', default=-1), -1)

    def test_longest_prefix_value(self):
        self.assertEqual(self.trie.longest_prefix_value('antonym'), 1)
        self.assertEqual(self.trie.longest_prefix_value('are'), 6)
        self.assertEqual(self.trie.longest_prefix_value('alla'), 2)
        self.assertEqual(self.trie.longest_prefix_value('allo'), 2)
        self.assertRaises(KeyError, self.trie.longest_prefix_value, 'alumni')
        self.assertEqual(self.trie.longest_prefix_value('alumni', default=None),
                         None)
        self.assertEqual(self.trie.longest_prefix_value('linux', default=-1),
                         -1)

    def test_longest_prefix_item(self):
        self.assertEqual(self.trie.longest_prefix_item('antonym'), ('ant', 1))
        self.assertEqual(self.trie.longest_prefix_item('are'), ('are', 6))
        self.assertEqual(self.trie.longest_prefix_item('alla'), ('all', 2))
        self.assertEqual(self.trie.longest_prefix_item('allo'), ('all', 2))
        self.assertRaises(KeyError, self.trie.longest_prefix_item, 'alumni')
        self.assertEqual(self.trie.longest_prefix_item('alumni', default=None),
                         None)
        self.assertEqual(self.trie.longest_prefix_item('linux', default=-1), -1)

    def test_iter_prefixes(self):
        self.assertEqual(list(self.trie.iter_prefixes('antonym')),
                         ['an', 'ant'])
        self.assertEqual(list(self.trie.iter_prefixes('are')), ['are'])
        self.assertEqual(list(self.trie.iter_prefixes('alumni')), [])

    def test_iter_prefix_values(self):
        self.assertEqual(list(self.trie.iter_prefix_values('antonym')), [0, 1])
        self.assertEqual(list(self.trie.iter_prefix_values('are')), [6])
        self.assertEqual(list(self.trie.iter_prefix_values('alumni')), [])

    def test_iter_prefix_items(self):
        self.assertEqual(list(self.trie.iter_prefix_items('antonym')),
                         [('an', 0), ('ant', 1)])
        self.assertEqual(list(self.trie.iter_prefix_items('are')), [('are', 6)])
        self.assertEqual(list(self.trie.iter_prefix_items('alumni')), [])

    def test_keys_wprefix(self):
        self.assertEqual(self.trie.keys('al'),
                         ['all', 'allot', 'alloy', 'aloe'])
        self.assertEqual(self.trie.keys('are'), ['are'])
        self.assertEqual(self.trie.keys('ann'), [])

    def test_values_wprefix(self):
        self.assertEqual(self.trie.values('al'), [2, 3, 4, 5])
        self.assertEqual(self.trie.values('are'), [6])
        self.assertEqual(self.trie.values('ann'), [])

    def test_items_wprefix(self):
        self.assertEqual(self.trie.items('al'),
                         [('all', 2), ('allot', 3), ('alloy', 4), ('aloe', 5)])
        self.assertEqual(self.trie.items('are'), [('are', 6)])
        self.assertEqual(self.trie.items('ann'), [])

    def test_consistency_wprefix(self):
        trie = self.trie
        for prefix in 'al', 'are', 'ann':
            self.assertEqual(
                trie.items(prefix),
                list(zip(trie.keys(prefix), trie.values(prefix)))
            )

    def test_empty_string(self):
        self.trie[''] = '!'

        self.assertEqual(self.trie.keys(''),
                         ['', 'all', 'allot', 'alloy', 'aloe', 'an', 'ant',
                          'are', 'ate', 'be'])
        self.assertEqual(self.trie.values(''),
                         ['!', 2, 3, 4, 5, 0, 1, 6, 7, 8])
        self.assertEqual(self.trie.items(''),
                         [('', '!'), ('all', 2), ('allot', 3), ('alloy', 4),
                          ('aloe', 5), ('an', 0), ('ant', 1), ('are', 6),
                          ('ate', 7), ('be', 8)])

        self.assertEqual(list(self.trie.iter_prefixes('foo')), [''])
        self.assertEqual(list(self.trie.iter_prefix_values('foo')), ['!'])
        self.assertEqual(list(self.trie.iter_prefix_items('foo')), [('', '!')])

        self.assertEqual(self.trie.longest_prefix('foo'), '')
        self.assertEqual(self.trie.longest_prefix_value('foo'), '!')
        self.assertEqual(self.trie.longest_prefix_item('foo'), ('', '!'))

    def test_pickle(self):
        from pickle import dumps, loads, HIGHEST_PROTOCOL
        for proto in range(HIGHEST_PROTOCOL):
            unpickled = loads(dumps(self.trie, proto))
            self.assertEqual(self.trie, unpickled)
            self.assertTrue(type(self.trie) is type(unpickled))
            self.assertTrue(self.trie is not unpickled)

    def test_repr(self):
        evaled = eval(repr(self.trie))
        self.assertEqual(evaled, self.trie)
        self.assertEqual(evaled.__class__, self.trie.__class__)

if __name__ == "__main__":
    unittest.main()    #启动测试

# unittest模块提供的方法：
# 方法	测试内容
# assertEqual(a, b)     	a == b
# assertNotEqual(a, b)       	a != b
# assertTrue(x)           	bool(x) is True
# assertFalse(x)         	bool(x) is False
# assertAlmostEqual(a, b)           	round(a-b, 7) == 0
# assertNotAlmostEqual(a, b)           	round(a-b, 7) != 0
# assertIs(a, b)	a is b
# assertIsNot(a, b)	a is not b
# assertIsNone(x)	x is None
# assertIsNotNone(x)	x is not None
# assertIn(a, b)	a in b
# assertNotIn(a, b)	a not in b
# assertIsInstance(a, b)	isinstance(a, b)
# assertNotIsInstance(a, b)	not isinstance(a, b)
# assertGreater(a, b)	a > b
# assertGreaterEqual(a, b)	a >= b
# assertLess(a, b)	a < b
# assertLessEqual(a, b)	a <= b
# assertRegexpMatches(s, re)	regex.search(s)
# assertNotRegexpMatches(s, re)	not regex.search(s)
# assertItemsEqual(a, b)	sorted(a) == sorted(b) and works with unhashable objs
# assertDictContainsSubset(a, b)	all the key/value pairs in a exist in b
# assertMultiLineEqual(a, b)	strings
# assertSequenceEqual(a, b)	sequences
# assertListEqual(a, b)	lists
# assertTupleEqual(a, b)	tuples
# assertSetEqual(a, b)	sets or frozensets
# assertDictEqual(a, b)	dicts
# assertMultiLineEqual(a, b) 	strings
# assertSequenceEqual(a, b) 	sequences
# assertListEqual(a, b)	lists
# assertTupleEqual(a, b)	tuples
# assertSetEqual(a, b) 	sets or frozensets
# assertDictEqual(a, b) 	dicts