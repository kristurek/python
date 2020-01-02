import unittest

from common import ListNode
from common import TreeNode
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

    def test_lengthOfLastWord_58(self):
        self.assertEqual(5, self.solution.lengthOfLastWord_58('Hello World'))

    def test_plusOne_66(self):
        self.assertEqual([1, 2, 4], self.solution.plusOne_66([1, 2, 3]))
        self.assertEqual([1, 0, 0, 0], self.solution.plusOne_66([9, 9, 9]))
        self.assertEqual([1, 2, 0], self.solution.plusOne_66([1, 1, 9]))

    def test_addBinary_67(self):
        self.assertEqual('10', self.solution.addBinary_67('1', '1'))
        self.assertEqual('110', self.solution.addBinary_67('11', '11'))
        self.assertEqual('101', self.solution.addBinary_67('10', '11'))

    def test_mySqrt_69(self):
        self.assertEqual(2, self.solution.mySqrt_69(4))
        self.assertEqual(2, self.solution.mySqrt_69(8))

    def test_climbStairs_70(self):
        self.assertEqual(0, self.solution.climbStairs_70(0))
        self.assertEqual(1, self.solution.climbStairs_70(1))
        self.assertEqual(2, self.solution.climbStairs_70(2))
        self.assertEqual(3, self.solution.climbStairs_70(3))
        self.assertEqual(5, self.solution.climbStairs_70(4))
        self.assertEqual(8, self.solution.climbStairs_70(5))

    def test_deleteDuplicates_83(self):
        head = ListNode(1)
        head.next = ListNode(1)
        head.next.next = ListNode(2)
        head.next.next.next = ListNode(3)
        head.next.next.next.next = ListNode(3)

        head = self.solution.deleteDuplicates_83(head)

        self.assertEqual(1, head.val)
        self.assertEqual(2, head.next.val)
        self.assertEqual(3, head.next.next.val)

    def test_merge_88(self):
        a = [1, 2, 3, 0, 0, 0]
        b = [1, 2, 3]
        self.solution.merge_88(a, 3, b, 3)
        self.assertEqual([1, 1, 2, 2, 3, 3], a)

    def test_isSameTree_100(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual(True, self.solution.isSameTree_100(root, root))

    def test_isSymmetric_101(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(2)
        root.left.left = TreeNode(3)
        root.left.right = TreeNode(4)
        root.right.left = TreeNode(4)
        root.right.right = TreeNode(3)

        self.assertEqual(True, self.solution.isSymmetric_101(root))

    def test_maxDepth_104(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(2)
        root.left.left = TreeNode(3)
        root.left.right = TreeNode(4)
        root.right.right = TreeNode(3)
        root.right.right.right = TreeNode(3)

        self.assertEqual(4, self.solution.maxDepth_104(root))

    def test_levelOrderBottom_107(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual([[4, 5, 6, 7], [2, 3], [1]], self.solution.levelOrderBottom_107(root))

    def test_sortedArrayToBST_108(self):
        root = self.solution.sortedArrayToBST_108([-10, -3, 0, 5, 9]);

        self.assertEqual(0, root.val)
        self.assertEqual(-3, root.left.val)
        self.assertEqual(9, root.right.val)
        self.assertEqual(-10, root.left.left.val)
        self.assertEqual(5, root.right.left.val)


if __name__ == '__main__':
    unittest.main()
