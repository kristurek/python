import unittest

from common import GraphNode
from common import ListNode
from common import Node3
from common import Node2
from common import TreeNode
from solution2 import Solution


class SolutionTest(unittest.TestCase):
    solution = Solution()

    def test_0_sample(self):
        self.assertTrue(self.solution._0_simple())

    def test_1_twoSum(self):
        self.assertCountEqual([0, 1], self.solution._1_twoSum([2, 7, 11, 15], 9))
        with self.assertRaises(ValueError):
            self.solution._1_twoSum([2, 7, 11, 15], 8)

    def test_2_addTwoNumbers(self):
        l1 = ListNode(2)
        l1.next = ListNode(4)
        l1.next.next = ListNode(3)

        l2 = ListNode(5)
        l2.next = ListNode(6)
        l2.next.next = ListNode(4)

        l3 = self.solution._2_addTwoNumbers(l1, l2)

        self.assertEqual(7, l3.val)
        self.assertEqual(0, l3.next.val)
        self.assertEqual(8, l3.next.next.val)

    def test_3_lengthOfLongestSubstring(self):
        self.assertEqual(3, self.solution._3_lengthOfLongestSubstring("abcabcbb"))
        self.assertEqual(1, self.solution._3_lengthOfLongestSubstring("bbbbb"))
        self.assertEqual(3, self.solution._3_lengthOfLongestSubstring("pwwkew"))

    def test_4_findMedianSortedArrays(self):
        self.assertEqual(2.0, self.solution._4_findMedianSortedArrays([1, 3], [2]))
        self.assertEqual(2.5, self.solution._4_findMedianSortedArrays([1, 2], [3, 4]))
        self.assertEqual(100000.5, self.solution._4_findMedianSortedArrays([100000], [100001]))

    def test_5_longestPalindrome(self):
        self.assertEqual("bb", self.solution._5_longestPalindrome("bb"))
        self.assertEqual("cc", self.solution._5_longestPalindrome("ccb"))
        self.assertEqual("cc", self.solution._5_longestPalindrome("acc"))
        self.assertEqual("bb", self.solution._5_longestPalindrome("cbbd"))
        self.assertEqual("bab", self.solution._5_longestPalindrome("babad"))
        self.assertEqual("a", self.solution._5_longestPalindrome("a"))
        self.assertEqual("a", self.solution._5_longestPalindrome("abc"))
        self.assertEqual("bb", self.solution._5_longestPalindrome("abb"))

        expected = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        param = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        self.assertEqual(expected, self.solution._5_longestPalindrome(param))

    def test_7_reverse(self):
        self.assertEqual(321, self.solution._7_reverse(123))
        self.assertEqual(-321, self.solution._7_reverse(-123))

    def test_9_isPalindrome(self):
        self.assertTrue(self.solution._9_isPalindrome(121))
        self.assertFalse(self.solution._9_isPalindrome(-121))
        self.assertFalse(self.solution._9_isPalindrome(123))

    def test_11_maxArea(self):
        self.assertEqual(49, self.solution._11_maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))

    def test_12_intToRoman(self):
        self.assertEqual("III", self.solution._12_intToRoman(3))
        self.assertEqual("IV", self.solution._12_intToRoman(4))
        self.assertEqual("IX", self.solution._12_intToRoman(9))
        self.assertEqual("LVIII", self.solution._12_intToRoman(58))
        self.assertEqual("MCMXCIV", self.solution._12_intToRoman(1994))

    def test_13_romanToInt(self):
        self.assertEqual(9, self.solution._13_romanToInt('IX'))
        self.assertEqual(4, self.solution._13_romanToInt('IV'))
        self.assertEqual(3, self.solution._13_romanToInt('III'))

    def test_14_longestCommonPrefix(self):
        self.assertEqual("fl", self.solution._14_longestCommonPrefix(["flower", "flow", "flight"]))
        self.assertEqual("", self.solution._14_longestCommonPrefix(["dog", "racecar", "car"]))

    def test_15_threeSum(self):
        expected = [[-1, 0, 1], [-1, -1, 2]]
        self.assertEqual(expected, self.solution._15_threeSum([-1, 0, 1, 2, -1, -4]))

        expected = [[0, 0, 0]]
        self.assertEqual(expected, self.solution._15_threeSum([0, 0, 0, 0]))

        expected = [[-2, 0, 2], [-2, 1, 1]]
        self.assertEqual(expected, self.solution._15_threeSum([-2, 0, 1, 1, 2]))

    def test_16_threeSumClosest(self):
        self.assertEqual(0, self.solution._16_threeSumClosest([-2, 0, 1, 1, 2], 0))
        self.assertEqual(2, self.solution._16_threeSumClosest([-1, 2, 1, -4], 1))
        self.assertEqual(0, self.solution._16_threeSumClosest([0, 2, 1, -3], 1))

    def test_17_letterCombinations(self):
        self.assertEqual(["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"],
                         self.solution._17_letterCombinations("23"))

    def test_19_removeNthFromEnd(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)

        head = self.solution._19_removeNthFromEnd(head, 3)

        self.assertEqual(2, head.val)
        self.assertEqual(3, head.next.val)

    def test_20_isValid(self):
        self.assertTrue(self.solution._20_isValid('()'))
        self.assertTrue(self.solution._20_isValid('()[]{}'))
        self.assertFalse(self.solution._20_isValid('(]'))
        self.assertFalse(self.solution._20_isValid('([)]'))
        self.assertTrue(self.solution._20_isValid('{[]}'))

    def test_21_mergeTwoLists(self):
        l1 = ListNode(1)
        l1.next = ListNode(2)
        l1.next.next = ListNode(3)

        l2 = ListNode(1)
        l2.next = ListNode(2)
        l2.next.next = ListNode(4)

        head = self.solution._21_mergeTwoLists(l1, l2)

        self.assertEqual(1, head.val)
        self.assertEqual(1, head.next.val)
        self.assertEqual(2, head.next.next.val)
        self.assertEqual(2, head.next.next.next.val)
        self.assertEqual(3, head.next.next.next.next.val)
        self.assertEqual(4, head.next.next.next.next.next.val)

    def test_22_generateParenthesis(self):
        expected = ["((()))", "(()())", "(())()", "()(())", "()()()"]
        self.assertEqual(expected, self.solution._22_generateParenthesis(3))

    def test_23_mergeKLists(self):
        head1 = ListNode(1)
        head1.next = ListNode(2)
        head1.next.next = ListNode(4)

        head2 = ListNode(1)
        head2.next = ListNode(3)
        head2.next.next = ListNode(4)

        head3 = ListNode(5)

        head = self.solution._23_mergeKLists([head1, head2, head3])

        self.assertEqual(1, head.val)
        self.assertEqual(1, head.next.val)
        self.assertEqual(2, head.next.next.val)
        self.assertEqual(3, head.next.next.next.val)
        self.assertEqual(4, head.next.next.next.next.val)
        self.assertEqual(4, head.next.next.next.next.next.val)
        self.assertEqual(5, head.next.next.next.next.next.next.val)

    def test_24_swapPairs(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)

        head = self.solution._24_swapPairs(head)

        self.assertEqual(2, head.val)
        self.assertEqual(1, head.next.val)
        self.assertEqual(3, head.next.next.val)

    def test_26_removeDuplicates(self):
        arr = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        length = self.solution._26_removeDuplicates(arr)
        self.assertEqual([0, 1, 2, 3, 4], arr[0:length])

    def test_27_removeElement(self):
        arr = [0, 0, 1, 1, 1, 2, 2, 3, 3, 1]
        length = self.solution._27_removeElement(arr, 1)
        self.assertEqual([0, 0, 2, 2, 3, 3], arr[0:length])

    def test_28_strStr(self):
        self.assertEqual(2, self.solution._28_strStr('Hello', 'll'))

    def test_33_search(self):
        self.assertEqual(4, self.solution._33_search([4, 5, 6, 7, 0, 1, 2], 0))
        self.assertEqual(-1, self.solution._33_search([4, 5, 6, 7, 0, 1, 2], 3))

    def test_34_searchRange(self):
        self.assertEqual([3, 4], self.solution._34_searchRange([5, 7, 7, 8, 8, 10], 8))
        self.assertEqual([-1, -1], self.solution._34_searchRange([5, 7, 7, 8, 8, 10], 6))

    def test_35_searchInsert(self):
        self.assertEqual(2, self.solution._35_searchInsert([2, 3, 5, 6, 7], 5))

    def test_36_isValidSudoku(self):
        param = [
            ["5", "3", ".", ".", "7", ".", ".", ".", "."],
            ["6", ".", ".", "1", "9", "5", ".", ".", "."],
            [".", "9", "8", ".", ".", ".", ".", "6", "."],
            ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
            ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
            ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
            [".", "6", ".", ".", ".", ".", "2", "8", "."],
            [".", ".", ".", "4", "1", "9", ".", ".", "5"],
            [".", ".", ".", ".", "8", ".", ".", "7", "9"]
        ]
        self.assertTrue(self.solution._36_isValidSudoku(param))

    def test_39_combinationSum(self):
        expected = [[2, 2, 3], [7]]
        self.assertEqual(expected, self.solution._39_combinationSum([2, 3, 6, 7], 7))

    def test_40_combinationSum2(self):
        expected = [[7]]
        self.assertEqual(expected, self.solution._40_combinationSum2([2, 3, 6, 7], 7))

    def test_41_firstMissingPositive(self):
        self.assertEqual(1, self.solution._41_firstMissingPositive([3, 7, 8, 9]))
        self.assertEqual(2, self.solution._41_firstMissingPositive([1, 3, 7, 8, 9]))
        self.assertEqual(4, self.solution._41_firstMissingPositive([1, 2, 3, 7, 8, 9]))
        self.assertEqual(2, self.solution._41_firstMissingPositive([-1, 1, 3, 4]))
        self.assertEqual(1, self.solution._41_firstMissingPositive([]))
        self.assertEqual(1, self.solution._41_firstMissingPositive([0]))

    def test_46_permute(self):
        expected = [[1, 2], [2, 1]]
        self.assertEqual(expected, self.solution._46_permute([1, 2]))

    def test_47_permuteUnique(self):
        expected = [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
        self.assertEqual(expected, self.solution._47_permuteUnique([1, 2, 1]))

    def test_49_groupAnagrams(self):
        expected = [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
        self.assertEqual(expected, self.solution._49_groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))

    def _disable_test_50_myPow(self):
        self.assertEqual(8, self.solution._50_myPow(2, 3))
        self.assertEqual(0, self.solution._50_myPow(2, -2147483648))

    def test_53_maxSubArray(self):
        self.assertEqual(6, self.solution._53_maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))

    def test_54_spiralOrder(self):
        self.assertEqual([1, 2, 3, 6, 9, 8, 7, 4, 5], self.solution._54_spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_58_lengthOfLastWord(self):
        self.assertEqual(5, self.solution._58_lengthOfLastWord('Hello World'))

    def test_61_rotateRight(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)

        head = self.solution._61_rotateRight(head, 2)

        self.assertEqual(3, head.val)
        self.assertEqual(4, head.next.val)
        self.assertEqual(1, head.next.next.val)
        self.assertEqual(2, head.next.next.next.val)
        self.assertIsNone(head.next.next.next.next)

    def test_66_plusOne(self):
        self.assertEqual([1, 2, 4], self.solution._66_plusOne([1, 2, 3]))
        self.assertEqual([1, 0, 0, 0], self.solution._66_plusOne([9, 9, 9]))
        self.assertEqual([1, 2, 0], self.solution._66_plusOne([1, 1, 9]))

    def test_67_addBinary(self):
        self.assertEqual('10', self.solution._67_addBinary('1', '1'))
        self.assertEqual('110', self.solution._67_addBinary('11', '11'))
        self.assertEqual('101', self.solution._67_addBinary('10', '11'))

    def test_69_mySqrt(self):
        self.assertEqual(2, self.solution._69_mySqrt(4))
        self.assertEqual(2, self.solution._69_mySqrt(8))

    def test_70_climbStairs(self):
        self.assertEqual(0, self.solution._70_climbStairs(0))
        self.assertEqual(1, self.solution._70_climbStairs(1))
        self.assertEqual(2, self.solution._70_climbStairs(2))
        self.assertEqual(3, self.solution._70_climbStairs(3))
        self.assertEqual(5, self.solution._70_climbStairs(4))
        self.assertEqual(8, self.solution._70_climbStairs(5))

    def test_71_simplifyPath(self):
        self.assertEqual("/home", self.solution._71_simplifyPath("/home/"))
        self.assertEqual("/", self.solution._71_simplifyPath("/../"))
        self.assertEqual("/home/foo", self.solution._71_simplifyPath("/home//foo/"))
        self.assertEqual("/c", self.solution._71_simplifyPath("/a/./b/../../c/"))
        self.assertEqual("/c", self.solution._71_simplifyPath("/a/../../b/../c//.//"))
        self.assertEqual("/a/b/c", self.solution._71_simplifyPath("/a//b////c/d//././/.."))

    def test_73_setZeroes(self):
        actual = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        expected = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
        self.solution._73_setZeroes(actual)

        self.assertEqual(expected, actual)

    def test_74_searchMatrix(self):
        self.assertTrue(self.solution._74_searchMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 7))

    def test_75_sortColors(self):
        nums = [2, 0, 2, 1, 1, 0]
        self.solution._75_sortColors(nums)

        self.assertEqual([0, 0, 1, 1, 2, 2], nums)

    def test_77_combine(self):
        expected = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

        self.assertEqual(expected, self.solution._77_combine(4, 2))

    def test_78_subsets(self):
        expected = [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]

        self.assertEqual(expected, self.solution._78_subsets([1, 2, 3]))

    def test_79_exist(self):
        self.assertTrue(self.solution._79_exist([
            ['A', 'B', 'C', 'E'],
            ['S', 'F', 'C', 'S'],
            ['A', 'D', 'E', 'E']
        ], "ABCCED"))
        self.assertTrue(self.solution._79_exist([
            ['A', 'B', 'C', 'E'],
            ['S', 'F', 'C', 'S'],
            ['A', 'D', 'E', 'E']
        ], "SEE"))
        self.assertFalse(self.solution._79_exist([
            ['A', 'B', 'C', 'E'],
            ['S', 'F', 'C', 'S'],
            ['A', 'D', 'E', 'E']
        ], "ABCB"))

    def test_80_removeDuplicates(self):
        nums = [1, 1, 1, 2, 2, 3]
        length = self.solution._80_removeDuplicates(nums)

        self.assertEqual(5, length)
        self.assertEqual(1, nums[0])
        self.assertEqual(1, nums[1])
        self.assertEqual(2, nums[2])
        self.assertEqual(2, nums[3])
        self.assertEqual(3, nums[4])

    def test_81_search(self):
        self.assertTrue(self.solution._81_search([2, 5, 6, 0, 0, 1, 2], 0))
        self.assertFalse(self.solution._81_search([2, 5, 6, 0, 0, 1, 2], 3))
        self.assertTrue(self.solution._81_search([1, 1, 3, 1], 3))
        self.assertTrue(self.solution._81_search([3, 1, 1], 3))
        self.assertTrue(self.solution._81_search([3, 5, 1], 1))

    def test_82_deleteDuplicates(self):
        head = ListNode(1)
        head.next = ListNode(1)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)

        head = self.solution._82_deleteDuplicates(head)

        self.assertEqual(3, head.val)
        self.assertEqual(4, head.next.val)
        self.assertIsNone(head.next.next)

    def test_83_deleteDuplicates(self):
        head = ListNode(1)
        head.next = ListNode(1)
        head.next.next = ListNode(2)
        head.next.next.next = ListNode(3)
        head.next.next.next.next = ListNode(3)

        head = self.solution._83_deleteDuplicates(head)

        self.assertEqual(1, head.val)
        self.assertEqual(2, head.next.val)
        self.assertEqual(3, head.next.next.val)
        self.assertIsNone(head.next.next.next)

    def test_86_partition(self):
        head = ListNode(1)
        head.next = ListNode(4)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(2)
        head.next.next.next.next = ListNode(5)
        head.next.next.next.next.next = ListNode(2)

        head = self.solution._86_partition(head, 3)

        self.assertEqual(1, head.val)
        self.assertEqual(2, head.next.val)
        self.assertEqual(2, head.next.next.val)
        self.assertEqual(4, head.next.next.next.val)
        self.assertEqual(3, head.next.next.next.next.val)
        self.assertEqual(5, head.next.next.next.next.next.val)
        self.assertIsNone(head.next.next.next.next.next.next)

    def test_88_merge(self):
        a = [1, 2, 3, 0, 0, 0]
        b = [1, 2, 3]
        self.solution._88_merge(a, 3, b, 3)
        self.assertEqual([1, 1, 2, 2, 3, 3], a)

    def test_90_subsetsWithDup(self):
        expected = [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]

        self.assertEqual(expected, self.solution._90_subsetsWithDup([2, 1, 2]))

    def test_92_reverseBetween(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)
        head.next.next.next.next = ListNode(5)
        head.next.next.next.next.next = ListNode(6)

        head = self.solution._92_reverseBetween(head, 3, 4)

        self.assertEqual(1, head.val)
        self.assertEqual(2, head.next.val)
        self.assertEqual(4, head.next.next.val)
        self.assertEqual(3, head.next.next.next.val)
        self.assertEqual(5, head.next.next.next.next.val)
        self.assertEqual(6, head.next.next.next.next.next.val)
        self.assertIsNone(head.next.next.next.next.next.next)

    def test_94_inorderTraversal(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual([4, 2, 5, 1, 6, 3, 7], self.solution._94_inorderTraversal(root))

    def test_98_isValidBST(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertFalse(self.solution._98_isValidBST(root))

    def test_100_isSameTree(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual(True, self.solution._100_isSameTree(root, root))

    def test_101_isSymmetric(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(2)
        root.left.left = TreeNode(3)
        root.left.right = TreeNode(4)
        root.right.left = TreeNode(4)
        root.right.right = TreeNode(3)

        self.assertEqual(True, self.solution._101_isSymmetric(root))

    def test_102_levelOrder(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        expected = [[1], [2, 3], [4, 5, 6, 7]]

        self.assertEqual(expected, self.solution._102_levelOrder(root))

    def test_103_zigzagLevelOrder(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        expected = [[1], [3, 2], [4, 5, 6, 7]]

        self.assertEqual(expected, self.solution._103_zigzagLevelOrder(root))

    def test_104_maxDepth(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(2)
        root.left.left = TreeNode(3)
        root.left.right = TreeNode(4)
        root.right.right = TreeNode(3)
        root.right.right.right = TreeNode(3)

        self.assertEqual(4, self.solution._104_maxDepth(root))

    def test_107_levelOrderBottom(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual([[4, 5, 6, 7], [2, 3], [1]], self.solution._107_levelOrderBottom(root))

    def test_108_sortedArrayToBST(self):
        root = self.solution._108_sortedArrayToBST([-10, -3, 0, 5, 9])

        self.assertEqual(0, root.val)
        self.assertEqual(-10, root.left.val)
        self.assertEqual(5, root.right.val)
        self.assertEqual(-3, root.left.right.val)
        self.assertEqual(9, root.right.right.val)

    def test_109_sortedListToBST(self):
        head = ListNode(-10)
        head.next = ListNode(-3)
        head.next.next = ListNode(0)
        head.next.next.next = ListNode(5)
        head.next.next.next.next = ListNode(9)

        root = self.solution._109_sortedListToBST(head)

        self.assertEqual(0, root.val)
        self.assertEqual(-10, root.left.val)
        self.assertEqual(5, root.right.val)
        self.assertEqual(-3, root.left.right.val)
        self.assertEqual(9, root.right.right.val)

    def test_110_isBalanced(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.right.right = TreeNode(7)

        self.assertTrue(self.solution._110_isBalanced(root))

        root.right.right.right = TreeNode(8)

        self.assertFalse(self.solution._110_isBalanced(root))

    def test_111_minDepth(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual(2, self.solution._111_minDepth(root))

    def test_112_hasPathSum(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual(True, self.solution._112_hasPathSum(root, 11))

    def test_113_pathSum(self):
        root = TreeNode(5)

        root.left = TreeNode(4)
        root.right = TreeNode(8)

        root.left.left = TreeNode(11)

        root.right.left = TreeNode(13)
        root.right.right = TreeNode(4)

        root.left.left.left = TreeNode(7)
        root.left.left.right = TreeNode(2)

        root.right.right.left = TreeNode(5)
        root.right.right.right = TreeNode(1)

        paths = self.solution._113_pathSum(root, 22)
        self.assertEqual(2, len(paths))
        self.assertEqual([5, 4, 11, 2], paths[0])
        self.assertEqual([5, 8, 4, 5], paths[1])

    def test_114_flatten(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(5)

        root.left.left = TreeNode(3)
        root.left.right = TreeNode(4)

        root.right.right = TreeNode(6)

        self.solution._114_flatten(root)

        self.assertEqual(1, root.val)
        self.assertEqual(2, root.right.val)
        self.assertEqual(3, root.right.right.val)
        self.assertEqual(4, root.right.right.right.val)
        self.assertEqual(5, root.right.right.right.right.val)
        self.assertEqual(6, root.right.right.right.right.right.val)

    def test_116_connect(self):
        root = Node3(1)

        root.left = Node3(2)
        root.right = Node3(3)

        root = self.solution._116_connect(root)

        self.assertIsNone(root.next)
        self.assertEqual(3, root.left.next.val)
        self.assertIsNone(root.right.next)

    def test_117_connect(self):
        root = Node3(1)

        root.left = Node3(2)
        root.right = Node3(3)

        root.left.left = Node3(4)
        root.right.right = Node3(5)

        root = self.solution._117_connect(root)

        self.assertIsNone(root.next)
        self.assertEqual(3, root.left.next.val)
        self.assertIsNone(root.right.next)
        self.assertEqual(5, root.left.left.next.val)

    def test_118_generate(self):
        results = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

        self.assertEqual(results, self.solution._118_generate(5))

    def test_119_getRow(self):
        results = [1, 4, 6, 4, 1]

        self.assertEqual(results, self.solution._119_getRow(4))

    def test_121_maxProfit(self):
        self.assertEqual(5, self.solution._121_maxProfit([7, 1, 5, 3, 6, 4]))
        self.assertEqual(0, self.solution._121_maxProfit([7, 6, 4, 3, 1]))

    def test_122_maxProfit(self):
        self.assertEqual(7, self.solution._122_maxProfit([7, 1, 5, 3, 6, 4]))
        self.assertEqual(4, self.solution._122_maxProfit([1, 2, 3, 4, 5]))
        self.assertEqual(0, self.solution._122_maxProfit([7, 6, 4, 3, 1]))

    def test_125_isPalindrome(self):
        self.assertEqual(True, self.solution._125_isPalindrome("A man, a plan, a canal: Panama"))
        self.assertEqual(False, self.solution._125_isPalindrome("race a car"))

    def test_129_sumNumbers(self):
        root = TreeNode(4)
        root.left = TreeNode(9)
        root.right = TreeNode(0)
        root.left.left = TreeNode(5)
        root.left.right = TreeNode(1)

        self.assertEqual(1026, self.solution._129_sumNumbers(root))

    def test_130_solve(self):
        actual = [['X', 'X', 'X', 'X'], ['X', 'O', 'O', 'X'], ['X', 'X', 'O', 'X'], ['X', 'O', 'X', 'X']]
        expected = [['X', 'X', 'X', 'X'], ['X', 'X', 'X', 'X'], ['X', 'X', 'X', 'X'], ['X', 'O', 'X', 'X']]

        self.solution._130_solve(actual)

        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertEqual(expected[2], actual[2])
        self.assertEqual(expected[3], actual[3])

    def test_131_partition(self):
        expected = [["a", "a", "b"], ["aa", "b"]]

        self.assertEqual(expected, self.solution._131_partition("aab"))

    def test_133_cloneGraph(self):
        n1 = GraphNode(1, [])
        n2 = GraphNode(2, [])
        n3 = GraphNode(3, [])
        n4 = GraphNode(4, [])

        n1.neighbors.append(n2)
        n1.neighbors.append(n4)

        n2.neighbors.append(n1)
        n2.neighbors.append(n3)

        n3.neighbors.append(n2)
        n3.neighbors.append(n4)

        n4.neighbors.append(n1)
        n4.neighbors.append(n3)

        actual = self.solution._133_cloneGraph(n1)

        self.assertEqual(1, actual.val)
        self.assertEqual(2, actual.neighbors[0].val)
        self.assertEqual(1, actual.neighbors[0].neighbors[0].val)
        self.assertEqual(3, actual.neighbors[0].neighbors[1].val)

    def test_136_singleNumber(self):
        self.assertEqual(1, self.solution._136_singleNumber([2, 2, 1]))
        self.assertEqual(4, self.solution._136_singleNumber([4, 1, 2, 1, 2]))

    def test_137_singleNumber(self):
        self.assertEqual(99, self.solution._137_singleNumber([0, 0, 0, 99]))
        self.assertEqual(99, self.solution._137_singleNumber([99, 0, 0, 0]))
        self.assertEqual(99, self.solution._137_singleNumber([99]))
        self.assertEqual(1, self.solution._137_singleNumber([0, 0, 0, 1, 2, 2, 2]))
        self.assertEqual(1, self.solution._137_singleNumber([3, 5, 3, 5, 3, 1, 2, 3, 5, 2]))
        self.assertEqual(1, self.solution._137_singleNumber([3, 5, 3, 5, 3, 1, 2, 3, 5, 2]))

    def test_138_copyRandomList(self):
        head = Node2(1)
        head.next = Node2(2)
        head.next.next = Node2(3)

        head.random = head.next.next

        actual = self.solution._138_copyRandomList(head)

        self.assertEqual(1, actual.val)
        self.assertEqual(2, actual.next.val)
        self.assertEqual(3, actual.next.next.val)
        self.assertEqual(3, actual.random.val)
        self.assertIsNone(actual.next.next.next)

    def test_139_wordBreak(self):
        self.assertTrue(self.solution._139_wordBreak("leetcode", ["leet", "code"]))

    def test_141_hasCycle(self):
        head = ListNode(0)
        head.next = ListNode(1)
        head.next.next = ListNode(2)
        head.next.next.next = ListNode(3)
        head.next.next.next.next = head.next

        self.assertEqual(True, self.solution._141_hasCycle(head))

    def test_142_detectCycle(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)
        head.next.next.next.next = head.next

        self.assertEqual(head.next, self.solution._142_detectCycle(head))

    def test_143_reorderList(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)
        head.next.next.next.next = ListNode(5)

        self.solution._143_reorderList(head)

        self.assertEqual(1, head.val)
        self.assertEqual(5, head.next.val)
        self.assertEqual(2, head.next.next.val)
        self.assertEqual(4, head.next.next.next.val)
        self.assertEqual(3, head.next.next.next.next.val)

        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)

        self.solution._143_reorderList(head)

        self.assertEqual(1, head.val)
        self.assertEqual(4, head.next.val)
        self.assertEqual(2, head.next.next.val)
        self.assertEqual(3, head.next.next.next.val)

    def test_144_preorderTraversal(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual([1, 2, 4, 5, 3, 6, 7], self.solution._144_preorderTraversal(root))

    def test_145_postorderTraversal(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        self.assertEqual([4, 5, 2, 6, 7, 3, 1], self.solution._145_postorderTraversal(root))

    def test_146_lruCache(self):
        cache = self.solution._146_lruCache(2)

        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(1, cache.get(1))
        cache.put(3, 3)  # evicts key 2
        self.assertEqual(-1, cache.get(2))  # returns -1 (not found)
        cache.put(4, 4)  # evicts key 1
        self.assertEqual(-1, cache.get(1))  # returns -1 (not found)
        self.assertEqual(3, cache.get(3))  # returns 3
        self.assertEqual(4, cache.get(4))  # returns 4


if __name__ == '__main__':
    unittest.main()
