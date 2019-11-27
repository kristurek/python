import unittest

from common import ListNode
from solution import Solution


class SolutionTest(unittest.TestCase):
    solution = Solution()

    def test_sample(self):
        self.assertTrue(self.solution.simple())

    def test_two_sum_1(self):
        self.assertCountEqual([0, 1], self.solution.two_sum_1([2, 7, 11, 15], 9))
        with self.assertRaises(ValueError):
            self.solution.two_sum_1([2, 7, 11, 15], 8)

    def test_reverse_7(self):
        self.assertEqual(-321, self.solution.reverse_7(-123))

    def test_isPalindrome_9(self):
        self.assertTrue(self.solution.isPalindrome_9(121))
        self.assertFalse(self.solution.isPalindrome_9(-121))
        self.assertFalse(self.solution.isPalindrome_9(123))

    def test_romanToInt_13(self):
        self.assertEqual(9, self.solution.romanToInt_13('IX'))
        self.assertEqual(4, self.solution.romanToInt_13('IV'))
        self.assertEqual(3, self.solution.romanToInt_13('III'))

    def test_longestCommonPrefix_14(self):
        self.assertEqual("fl", self.solution.longestCommonPrefix_14(["flower", "flow", "flight"]))
        self.assertEqual("", self.solution.longestCommonPrefix_14(["dog", "racecar", "car"]))

    def test_isValid_20(self):
        self.assertTrue(self.solution.isValid_20('()'))
        self.assertTrue(self.solution.isValid_20('()[]{}'))
        self.assertFalse(self.solution.isValid_20('(]'))
        self.assertFalse(self.solution.isValid_20('([)]'))
        self.assertTrue(self.solution.isValid_20('{[]}'))

    def test_mergeTwoLists_21(self):
        l1 = ListNode(1)
        l1.next = ListNode(2)
        l1.next.next = ListNode(3)

        l2 = ListNode(1)
        l2.next = ListNode(2)
        l2.next.next = ListNode(4)

        head = self.solution.mergeTwoLists_21(l1, l2)

        self.assertEqual(1, head.val)
        self.assertEqual(1, head.next.val)
        self.assertEqual(2, head.next.next.val)
        self.assertEqual(2, head.next.next.next.val)
        self.assertEqual(3, head.next.next.next.next.val)
        self.assertEqual(4, head.next.next.next.next.next.val)

    def test_removeDuplicates_26(self):
        arr = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        length = self.solution.removeDuplicates_26(arr)
        self.assertEqual([0, 1, 2, 3, 4], arr[0:length])

    def test_removeElement_27(self):
        arr = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        length = self.solution.removeElement_27(arr, 1)
        self.assertEqual([0, 0, 2, 2, 3, 3, 4], arr[0:length])

    def test_strStr_28(self):
        self.assertEqual(2, self.solution.strStr_28('Hello', 'll'))

    def test_searchInsert_35(self):
        self.assertEqual(2, self.solution.searchInsert_35([2, 3, 5, 6, 7], 5))

    def test_maxSubArray_53(self):
        self.assertEqual(6, self.solution.maxSubArray_53([-2, 1, -3, 4, -1, 2, 1, -5, 4]))


if __name__ == '__main__':
    unittest.main()
