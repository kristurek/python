import unittest

from common import Employee, Node2, GraphNode, Node3
from common import ListNode
from common import Node
from common import TreeNode
from solution import Solution, Iterator


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
        self.assertEqual("aba", self.solution._5_longestPalindrome("babad"))
        self.assertEqual("a", self.solution._5_longestPalindrome("a"))
        self.assertEqual("c", self.solution._5_longestPalindrome("abc"))
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

    def test_18_fourSum(self):
        expected = [[-2, -1, 1, 2], [-1, 0, 0, 1], [-2, 0, 0, 2]]
        self.assertEqual(expected, self.solution._18_fourSum([1, 0, -1, 0, -2, 2], 0))

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

    def test_43_multiply(self):
        self.assertEqual("56088", self.solution._43_multiply("123", "456"))

    def test_46_permute(self):
        expected = [[1, 2], [2, 1]]
        self.assertEqual(expected, self.solution._46_permute([1, 2]))

    def test_47_permuteUnique(self):
        expected = [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
        self.assertEqual(expected, self.solution._47_permuteUnique([1, 2, 1]))

    def test_49_groupAnagrams(self):
        expected = [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
        self.assertEqual(expected, self.solution._49_groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))

    def test_50_myPow(self):
        self.assertEqual(8, self.solution._50_myPow_v2(2, 3))
        self.assertEqual(0, self.solution._50_myPow_v2(2, -2147483648))

    def test_53_maxSubArray(self):
        self.assertEqual(6, self.solution._53_maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))

    def test_54_spiralOrder(self):
        self.assertEqual([1, 2, 3, 6, 9, 8, 7, 4, 5], self.solution._54_spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_58_lengthOfLastWord(self):
        self.assertEqual(5, self.solution._58_lengthOfLastWord('Hello World'))

    def test_60_getPermutation(self):
        self.assertEqual("213", self.solution._60_getPermutation(3, 3))

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
        answer = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        expected = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
        self.solution._73_setZeroes(answer)

        self.assertEqual(expected, answer)

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

    def test_105_buildTree(self):
        root = TreeNode(3)

        root.left = TreeNode(9)
        root.right = TreeNode(20)

        root.right.left = TreeNode(15)
        root.right.right = TreeNode(7)

        actual = self.solution._105_buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])

        self.assertEqual(root.val, actual.val)
        self.assertEqual(root.left.val, actual.left.val)
        self.assertEqual(root.right.val, actual.right.val)
        self.assertEqual(root.right.left.val, actual.right.left.val)
        self.assertEqual(root.right.right.val, actual.right.right.val)

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

    def test_148_sortList(self):
        head = ListNode(4)
        head.next = ListNode(2)
        head.next.next = ListNode(1)
        head.next.next.next = ListNode(3)

        head = self.solution._148_sortList(head)

        self.assertEqual(1, head.val)
        self.assertEqual(2, head.next.val)
        self.assertEqual(3, head.next.next.val)
        self.assertEqual(4, head.next.next.next.val)
        self.assertIsNone(head.next.next.next.next)

    def test_151_reverseWords(self):
        self.assertEqual("example good a", self.solution._151_reverseWords("a good   example"))

    def test_153_findMin(self):
        self.assertEqual(8, self.solution._153_findMin([8, 9]))
        self.assertEqual(1, self.solution._153_findMin([3, 4, 5, 1, 2]))
        self.assertEqual(0, self.solution._153_findMin([4, 5, 6, 7, 0, 1, 2]))

    def test_155_minStack(self):
        minStack = self.solution._155_minStack()
        minStack.push(-2)
        minStack.push(0)
        minStack.push(-3)
        self.assertEqual(-3, minStack.getMin())
        minStack.pop()
        self.assertEqual(0, minStack.top())
        self.assertEqual(-2, minStack.getMin())

    # fully tested on LeetCode but locally not working
    # FIXME
    def disabled_test_160_getIntersectionNode(self):
        head1 = ListNode(1)
        head1.next = ListNode(2)
        head1.next.next = ListNode(3)
        head1.next.next.next = ListNode(4)

        head2 = ListNode(7)
        head2.next = ListNode(6)
        head2.next.next = ListNode(3)
        head2.next.next.next = ListNode(4)

        head3 = self.solution._160_getIntersectionNode(head1, head2)

        self.assertEqual(3, head3.val)
        self.assertEqual(4, head3.next.val)
        self.assertIsNone(head3.next.next.next)

    def test_165_compareVersion(self):
        self.assertEqual(-1, self.solution._165_compareVersion("0.1", "1.1"))
        self.assertEqual(1, self.solution._165_compareVersion("1.0.1", "1"))
        self.assertEqual(-1, self.solution._165_compareVersion("7.5.2.4", "7.5.3"))
        self.assertEqual(0, self.solution._165_compareVersion("1.01", "1.001"))
        self.assertEqual(0, self.solution._165_compareVersion("1.0", "1.0.0"))

    def test_167_twoSum(self):
        self.assertCountEqual([1, 2], self.solution._167_twoSum([2, 7, 11, 15], 9))
        with self.assertRaises(ValueError):
            self.solution._167_twoSum([2, 7, 11, 15], 8)

    def test_168_convertToTitle(self):
        self.assertEqual("A", self.solution._168_convertToTitle(1))
        self.assertEqual("Z", self.solution._168_convertToTitle(26))
        self.assertEqual("AA", self.solution._168_convertToTitle(27))
        self.assertEqual("AMJ", self.solution._168_convertToTitle(1024))

    def test__169_majorityElement(self):
        self.assertEqual(3, self.solution._169_majorityElement([3, 2, 3]))
        self.assertEqual(2, self.solution._169_majorityElement([2, 2, 1, 1, 1, 2, 2]))

    def test_171_titleToNumber(self):
        self.assertEqual(1, self.solution._171_titleToNumber("A"))
        self.assertEqual(26, self.solution._171_titleToNumber("Z"))
        self.assertEqual(27, self.solution._171_titleToNumber("AA"))
        self.assertEqual(1024, self.solution._171_titleToNumber("AMJ"))

    def test_172_trailingZeroes(self):
        self.assertEqual(0, self.solution._172_trailingZeroes(0))
        self.assertEqual(0, self.solution._172_trailingZeroes(4))
        self.assertEqual(1, self.solution._172_trailingZeroes(5))
        self.assertEqual(2, self.solution._172_trailingZeroes(10))
        self.assertEqual(7, self.solution._172_trailingZeroes(30))

    def test_173_BSTIterator(self):
        root = TreeNode(7)
        root.left = TreeNode(3)
        root.right = TreeNode(15)
        root.right.left = TreeNode(9)
        root.right.right = TreeNode(20)

        iterator = self.solution._173_BSTIterator(root)
        self.assertEqual(3, iterator.next())
        self.assertEqual(7, iterator.next())
        self.assertTrue(iterator.hasNext())
        self.assertEqual(9, iterator.next())
        self.assertTrue(iterator.hasNext())
        self.assertEqual(15, iterator.next())
        self.assertTrue(iterator.hasNext())
        self.assertEqual(20, iterator.next())
        self.assertFalse(iterator.hasNext())

    def test_187_findRepeatedDnaSequences(self):
        actual = self.solution._187_findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")
        expected = ["CCCCCAAAAA", "AAAAACCCCC"]

        self.assertEqual(sorted(expected), sorted(actual))

    def test_189_rotate(self):
        nums = [1, 2, 3, 4, 5, 6, 7]
        self.solution._189_rotate(nums, 3)
        self.assertEqual([5, 6, 7, 1, 2, 3, 4], nums)

    def test_198_rob(self):
        self.assertEqual(4, self.solution._198_rob([1, 2, 3, 1]))
        self.assertEqual(12, self.solution._198_rob([2, 7, 9, 3, 1]))

    def test_199_rightSideView(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)

        actual = self.solution._199_rightSideView(root)

        self.assertEqual(1, actual[0])
        self.assertEqual(3, actual[1])
        self.assertEqual(4, actual[2])

    def test_200_numIslands(self):
        self.assertEqual(1, self.solution._200_numIslands(
            [['1', '1', '1', '1', '0'], ['1', '1', '0', '1', '0'], ['1', '1', '0', '0', '0'],
             ['0', '0', '0', '0', '0']]))
        self.assertEqual(3, self.solution._200_numIslands(
            [['1', '1', '0', '0', '0'], ['1', '1', '0', '0', '0'], ['0', '0', '1', '0', '0'],
             ['0', '0', '0', '1', '1']]))

    def test_201_rangeBitwiseAnd(self):
        self.assertEqual(4, self.solution._201_rangeBitwiseAnd(5, 7))

    def test_202_isHappy(self):
        self.assertTrue(self.solution._202_isHappy(19))
        self.assertFalse(self.solution._202_isHappy(2))

    def test_203_removeElements(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(1)

        head = self.solution._203_removeElements(head, 1)

        self.assertEqual(2, head.val)
        self.assertEqual(3, head.next.val)
        self.assertIsNone(head.next.next)

    def test_205_isIsomorphic(self):
        self.assertTrue(self.solution._205_isIsomorphic("egg", "add"))
        self.assertFalse(self.solution._205_isIsomorphic("foo", "bar"))
        self.assertTrue(self.solution._205_isIsomorphic("paper", "title"))
        self.assertFalse(self.solution._205_isIsomorphic("ab", "aa"))

    def test_206_reverseList(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)

        actual = self.solution._206_reverseList(head)

        self.assertEqual(4, actual.val)
        self.assertEqual(3, actual.next.val)
        self.assertEqual(2, actual.next.next.val)
        self.assertEqual(1, actual.next.next.next.val)
        self.assertIsNone(actual.next.next.next.next)

    def test_208_trie(self):
        trie = self.solution._208_trie()

        trie.insert("apple")
        self.assertTrue(trie.search("apple"))
        self.assertFalse(trie.search("app"))
        self.assertTrue(trie.startsWith("app"))
        trie.insert("app")
        self.assertTrue(trie.search("app"))

    def test_211_wordDictionary(self):
        wordDictionary = self.solution._211_wordDictionary()

        wordDictionary.addWord("bad")
        wordDictionary.addWord("dad")
        wordDictionary.addWord("mad")

        self.assertFalse(wordDictionary.search("pad"))
        self.assertTrue(wordDictionary.search("bad"))
        self.assertTrue(wordDictionary.search(".ad"))
        self.assertTrue(wordDictionary.search("..."))

    def test_213_rob(self):
        self.assertEqual(0, self.solution._213_rob([]))
        self.assertEqual(2, self.solution._213_rob([2]))
        self.assertEqual(2, self.solution._213_rob([1, 2]))
        self.assertEqual(3, self.solution._213_rob([2, 3, 2]))
        self.assertEqual(4, self.solution._213_rob([1, 2, 3, 1]))

    def test_215_findKthLargest(self):
        self.assertEqual(5, self.solution._215_findKthLargest([3, 2, 1, 5, 6, 4], 2))
        self.assertEqual(4, self.solution._215_findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))

    def test_216_combinationSum3(self):
        actual = self.solution._216_combinationSum3(3, 7)

        self.assertEqual(1, len(actual))
        self.assertEqual([1, 2, 4], actual[0])

    def test_217_containsDuplicate(self):
        self.assertFalse(self.solution._217_containsDuplicate(None))
        self.assertFalse(self.solution._217_containsDuplicate([]))
        self.assertTrue(self.solution._217_containsDuplicate([1, 2, 3, 1]))
        self.assertFalse(self.solution._217_containsDuplicate([1, 2, 3, 4]))
        self.assertTrue(self.solution._217_containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))

    def test_219_containsNearbyDuplicate(self):
        self.assertTrue(self.solution._219_containsNearbyDuplicate([1, 2, 3, 1], 3))
        self.assertTrue(self.solution._219_containsNearbyDuplicate([1, 0, 1, 1], 1))
        self.assertFalse(self.solution._219_containsNearbyDuplicate([1, 2, 3, 1, 2, 3], 2))

    def test_225_myStack(self):
        stack = self.solution._225_MyStack()

        stack.push(1)
        stack.push(2)
        self.assertEqual(2, stack.top())
        self.assertEqual(2, stack.pop())
        self.assertFalse(stack.empty())

    def test_226_invertTree(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        root = self.solution._226_invertTree(root)

        self.assertEqual(1, root.val)
        self.assertEqual(3, root.left.val)
        self.assertEqual(2, root.right.val)

        self.assertEqual(7, root.left.left.val)
        self.assertEqual(6, root.left.right.val)
        self.assertEqual(5, root.right.left.val)
        self.assertEqual(4, root.right.right.val)

    def test_229_majorityElement(self):
        self.assertEqual([3], self.solution._229_majorityElement([3, 2, 3]))
        self.assertEqual([1, 2], self.solution._229_majorityElement([1, 1, 1, 3, 3, 2, 2, 2]))

    def test_230_kthSmallest(self):
        root = TreeNode(3)

        root.left = TreeNode(2)
        root.right = TreeNode(5)

        root.left.left = TreeNode(1)

        root.right.left = TreeNode(4)
        root.right.right = TreeNode(6)

        self.assertEqual(1, self.solution._230_kthSmallest(root, 1))
        self.assertEqual(2, self.solution._230_kthSmallest(root, 2))
        self.assertEqual(3, self.solution._230_kthSmallest(root, 3))

    def test_231_isPowerOfTwo(self):
        self.assertTrue(self.solution._231_isPowerOfTwo(1))
        self.assertTrue(self.solution._231_isPowerOfTwo(16))
        self.assertFalse(self.solution._231_isPowerOfTwo(218))

    def test_232_MyQueue(self):
        queue = self.solution._232_MyQueue()

        queue.push(1)
        queue.push(2)
        self.assertEqual(1, queue.peek())
        self.assertEqual(1, queue.pop())
        self.assertFalse(queue.empty())

    def test_234_isPalindrome(self):
        head = ListNode(1)
        head.next = ListNode(2)
        self.assertFalse(self.solution._234_isPalindrome(head))

        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(2)
        head.next.next.next = ListNode(1)
        self.assertTrue(self.solution._234_isPalindrome(head))

    def test_237_deleteNode(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)

        self.solution._237_deleteNode(head.next)

        self.assertEqual(1, head.val)
        self.assertEqual(3, head.next.val)
        self.assertEqual(4, head.next.next.val)
        self.assertIsNone(head.next.next.next)

    def test_238_productExceptSelf(self):
        self.assertEqual([24, 12, 8, 6], self.solution._238_productExceptSelf([1, 2, 3, 4]))

    def test_257_binaryTreePaths(self):
        root = TreeNode(1)

        root.left = TreeNode(2)
        root.right = TreeNode(3)

        root.left.right = TreeNode(5)

        actual = self.solution._257_binaryTreePaths(root)
        expected = ["1->2->5", "1->3"]

        self.assertEqual(expected, actual)

    def test_260_singleNumber(self):
        self.assertEqual([3, 5], self.solution._260_singleNumber([1, 2, 1, 3, 2, 5]))

    def test_264_nthUglyNumber(self):
        self.assertEqual(12, self.solution._264_nthUglyNumber(10))

    def test_268_missingNumber(self):
        self.assertEqual(1, self.solution._268_missingNumber([0]))
        self.assertEqual(2, self.solution._268_missingNumber([3, 0, 1]))
        self.assertEqual(8, self.solution._268_missingNumber([9, 6, 4, 2, 3, 5, 7, 0, 1]))

    def test_283_moveZeroes(self):
        nums = [0, 1, 0, 3, 12]
        answer = [1, 3, 12, 0, 0]
        self.solution._283_moveZeroes(nums)

        self.assertEqual(answer, nums)

    def test_284_PeekingIterator(self):
        nums = [1, 2, 3]
        iterator = self.solution._284_PeekingIterator(Iterator(nums))

        self.assertTrue(iterator.hasNext())
        self.assertEqual(1, iterator.peek())
        self.assertEqual(1, iterator.next())

        self.assertEqual(2, iterator.peek())
        self.assertEqual(2, iterator.next())

        self.assertEqual(3, iterator.next())
        self.assertFalse(iterator.hasNext())
        self.assertIsNone(iterator.peek())

    def test_287_findDuplicate(self):
        self.assertEqual(2, self.solution._287_findDuplicate([1, 3, 4, 2, 2]))

    def test_313_nthSuperUglyNumber(self):
        self.assertEqual(12, self.solution._313_nthSuperUglyNumber(10, [2, 3, 5]))
        self.assertEqual(1, self.solution._313_nthSuperUglyNumber(12, [1]))
        self.assertEqual(32, self.solution._313_nthSuperUglyNumber(12, [2, 7, 13, 19]))

    def test_318_maxProduct(self):
        self.assertEqual(16, self.solution._318_maxProduct(["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]))
        self.assertEqual(4, self.solution._318_maxProduct(["a", "ab", "abc", "d", "cd", "bcd", "abcd"]))
        self.assertEqual(0, self.solution._318_maxProduct(["a", "aa", "aaa", "aaaa"]))

    def test_344_reverseString(self):
        actual = ['h', 'e', 'l', 'l', 'o']
        expected = ['o', 'l', 'l', 'e', 'h']

        self.solution._344_reverseString(actual)

        self.assertEqual(expected, actual)

    def test_345_reverseVowels(self):
        self.assertEqual("holle", self.solution._345_reverseVowels("hello"))

    def test_349_intersection(self):
        self.assertEqual([9, 4], self.solution._349_intersection([4, 9, 5], [9, 4, 9, 8, 4]))

    def test_350_intersect(self):
        self.assertEqual([9, 4], self.solution._350_intersect([4, 9, 5], [9, 4, 9, 8, 4]))

    def test_387_firstUniqChar(self):
        self.assertEqual(0, self.solution._387_firstUniqChar("hello"))
        self.assertEqual(2, self.solution._387_firstUniqChar("loveleetcode"))

    def test_389_findTheDifference(self):
        self.assertEqual('e', self.solution._389_findTheDifference("abcd", "abced"))

    def test_392_isSubsequence(self):
        self.assertTrue(self.solution._392_isSubsequence("abc", "ahbgdc"))
        self.assertFalse(self.solution._392_isSubsequence("axc", "ahbgdc"))

    def test_429_levelOrder(self):
        root = Node(1, [Node(), Node(), Node()])
        root.children[0].val = 3
        root.children[1].val = 2
        root.children[2].val = 4
        root.children[0].children = [Node(), Node()]
        root.children[0].children[0].val = 5
        root.children[0].children[1].val = 6

        expected = [[1], [3, 2, 4], [5, 6]]

        self.assertEqual(expected, self.solution._429_levelOrder(root))

    def test_509_fib(self):
        self.assertEqual(0, self.solution._509_fib(0))
        self.assertEqual(1, self.solution._509_fib(1))
        self.assertEqual(1, self.solution._509_fib(2))
        self.assertEqual(2, self.solution._509_fib(3))
        self.assertEqual(3, self.solution._509_fib(4))

    def test_525_findMaxLength(self):
        self.assertEqual(2, self.solution._525_findMaxLength([0, 1]))
        self.assertEqual(2, self.solution._525_findMaxLength([0, 1, 0]))

    def test_543_diameterOfBinaryTree(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)

        self.assertEqual(3, self.solution._543_diameterOfBinaryTree(root))

    def test_557_reverseWords(self):
        self.assertEqual("s'teL ekat edoCteeL tsetnoc", self.solution._557_reverseWords("Let's take LeetCode contest"))

    def test_560_subarraySum(self):
        self.assertEqual(2, self.solution._560_subarraySum([1, 1, 1], 2))
        self.assertEqual(3, self.solution._560_subarraySum([0, 1, 1, 2], 2))

    def test_561_arrayPairSum(self):
        self.assertEqual(4, self.solution._561_arrayPairSum([1, 4, 3, 2]))

    def test_589_preorder(self):
        root = Node(1, [Node(3), Node(2), Node(4)])
        root.children[0].children = [Node(5), Node(6)]

        answer = [1, 3, 5, 6, 2, 4]

        self.assertEqual(answer, self.solution._589_preorder(root))

    def test_590_postorder(self):
        root = Node(1, [Node(3), Node(2), Node(4)])
        root.children[0].children = [Node(5), Node(6)]

        answer = [5, 6, 3, 2, 4, 1]

        self.assertEqual(answer, self.solution._590_postorder(root))

    def test_617_mergeTrees(self):
        root1 = TreeNode(1)
        root1.left = TreeNode(2)
        root1.right = TreeNode(3)
        root1.left.left = TreeNode(4)
        root1.left.right = TreeNode(5)

        root2 = TreeNode(5)
        root2.left = TreeNode(6)
        root2.right = TreeNode(7)
        root2.right.right = TreeNode(8)

        root3 = self.solution._617_mergeTrees(root1, root2)

        self.assertEqual(6, root3.val)
        self.assertEqual(8, root3.left.val)
        self.assertEqual(10, root3.right.val)
        self.assertEqual(4, root3.left.left.val)
        self.assertEqual(5, root3.left.right.val)
        self.assertIsNone(root3.right.left)
        self.assertEqual(8, root3.right.right.val)

    def test_657_judgeCircle(self):
        self.assertTrue(self.solution._657_judgeCircle("UD"))
        self.assertFalse(self.solution._657_judgeCircle("LL"))

    def test_665_checkPossibility(self):
        self.assertTrue(self.solution._665_checkPossibility([]))
        self.assertTrue(self.solution._665_checkPossibility([3]))
        self.assertTrue(self.solution._665_checkPossibility([4, 2, 3]))
        self.assertTrue(self.solution._665_checkPossibility([2, 3, 3, 2]))
        self.assertFalse(self.solution._665_checkPossibility([4, 2, 1]))
        self.assertTrue(self.solution._665_checkPossibility([-1, 4, 2, 3]))

    def test_678_checkValidString(self):
        self.assertTrue(self.solution._678_checkValidString("()"))
        self.assertTrue(self.solution._678_checkValidString("(*)"))
        self.assertTrue(self.solution._678_checkValidString("(*))"))

    def test_690_getImportance(self):
        # [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
        workers = []

        e1 = Employee(1, 5, [2, 3])
        e2 = Employee(2, 3, [])
        e3 = Employee(3, 3, [])

        workers.append(e1)
        workers.append(e2)
        workers.append(e3)

        self.assertEqual(11, self.solution._690_getImportance(workers, 1))

    def test_700_searchBST(self):
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(7)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)

        self.assertEqual(root.left, self.solution._700_searchBST(root, 2))

    def test_705_myHashSet(self):
        myHashSet = self.solution._705_myHashSet()

        myHashSet.add(3)
        self.assertTrue(myHashSet.contains(3))
        myHashSet.remove(3)
        self.assertFalse(myHashSet.contains(3))
        myHashSet.add(3)
        myHashSet.add(9)
        self.assertTrue(myHashSet.contains(3))
        self.assertFalse(myHashSet.contains(1))

    def test_709_toLowerCase(self):
        self.assertEqual("", self.solution._709_toLowerCase(""))
        self.assertIsNone(self.solution._709_toLowerCase(None))
        self.assertEqual("123!~", self.solution._709_toLowerCase("123!~"))
        self.assertEqual("abc", self.solution._709_toLowerCase("abc"))
        self.assertEqual("abc", self.solution._709_toLowerCase("ABC"))
        self.assertEqual("abcd", self.solution._709_toLowerCase("aBcD"))

    def test_728_selfDividingNumbers(self):
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
        self.assertEqual(expected, self.solution._728_selfDividingNumbers(1, 22))

    def test_771_numJewelsInStones(self):
        self.assertEqual(3, self.solution._771_numJewelsInStones("aA", "aAAbbbb"))

    def test_796_rotateString(self):
        self.assertTrue(self.solution._796_rotateString("abcde", "cdeab"))
        self.assertFalse(self.solution._796_rotateString("abcde", "abced"))

    def test_804_uniqueMorseRepresentations(self):
        self.assertEqual(2, self.solution._804_uniqueMorseRepresentations(["gin", "zen", "gig", "msg"]))

    def test_811_subdomainVisits(self):
        expected = ["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]

        self.assertCountEqual(expected, self.solution._811_subdomainVisits(["9001 discuss.leetcode.com"]))

        expected = ["901 mail.com", "50 yahoo.com", "900 google.mail.com", "5 wiki.org", "5 org",
                    "1 intel.mail.com", "951 com"]

        self.assertCountEqual(expected, self.solution._811_subdomainVisits(
            ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]))

    def test_821_shortestToChar(self):
        expected = [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]

        self.assertEqual(expected, self.solution._821_shortestToChar("loveleetcode", 'e'))

    def test_832_flipAndInvertImage(self):
        self.assertEqual([[1, 0, 0]], self.solution._832_flipAndInvertImage([[1, 1, 0]]))
        self.assertEqual([[1, 0, 0], [0, 1, 0], [1, 1, 1]],
                         self.solution._832_flipAndInvertImage([[1, 1, 0], [1, 0, 1], [0, 0, 0]]))

    def test_844_backspaceCompare(self):
        self.assertTrue(self.solution._844_backspaceCompare("", ""))
        self.assertTrue(self.solution._844_backspaceCompare("ab#c", "ad#c"))
        self.assertTrue(self.solution._844_backspaceCompare("ab##", "c#d#"))
        self.assertTrue(self.solution._844_backspaceCompare("a##c", "#a#c"))
        self.assertFalse(self.solution._844_backspaceCompare("a#c", "b"))

    def test_852_peakIndexInMountainArray(self):
        self.assertEqual(1, self.solution._852_peakIndexInMountainArray([0, 1, 0]))
        self.assertEqual(1, self.solution._852_peakIndexInMountainArray([0, 2, 1, 0]))

    def test_876_middleNode(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)

        self.assertEqual(3, self.solution._876_middleNode(head).val)

    def test_905_sortArrayByParity(self):
        self.assertEqual([2, 4, 3, 1], self.solution._905_sortArrayByParity([1, 2, 3, 4]))

    def test_922_sortArrayByParityII(self):
        self.assertEqual([4, 5, 2, 7], self.solution._922_sortArrayByParityII([4, 2, 5, 7]))

    def test_929_numUniqueEmails(self):
        self.assertEqual(2, self.solution._929_numUniqueEmails(
            ["test.email+alex@leetcode.com", "test.e.mail+bob.cathy@leetcode.com", "testemail+david@lee.tcode.com"]))

        self.assertEqual(1, self.solution._929_numUniqueEmails(
            ["test.email+alex@leetcode.com", "test.email@leetcode.com"]))

    # FIXME
    def test_937_reorderLogFiles(self):
        expected = ["let1 art can", "let3 art zero", "let2 own kit dig", "dig1 8 1 5 1", "dig2 3 6"]

        self.assertEqual(expected, self.solution._937_reorderLogFiles(
            ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]))

    def test_938_rangeSumBST(self):
        root = TreeNode(10)
        root.left = TreeNode(5)
        root.right = TreeNode(15)
        root.left.left = TreeNode(3)
        root.left.right = TreeNode(7)
        root.right.right = TreeNode(18)

        self.assertEqual(32, self.solution._938_rangeSumBST(root, 7, 15))

    def test_942_diStringMatch(self):
        self.assertEqual([0, 4, 1, 3, 2], self.solution._942_diStringMatch("IDID"))
        self.assertEqual([3, 2, 0, 1], self.solution._942_diStringMatch("DDI"))
        self.assertEqual([0, 1, 2, 3], self.solution._942_diStringMatch("III"))
        self.assertEqual([3, 2, 1, 0], self.solution._942_diStringMatch("DDD"))

    def test_944_minDeletionSize(self):
        self.assertEqual(1, self.solution._944_minDeletionSize(["cba", "daf", "ghi"]))
        self.assertEqual(2, self.solution._944_minDeletionSize(["rrjk", "furt", "guzm"]))
        self.assertEqual(3, self.solution._944_minDeletionSize(["zyx", "wvu", "tsr"]))

    def test_961_repeatedNTimes(self):
        self.assertEqual(3, self.solution._961_repeatedNTimes([1, 2, 3, 3]))
        self.assertEqual(2, self.solution._961_repeatedNTimes([2, 1, 2, 5, 3, 2]))
        self.assertEqual(5, self.solution._961_repeatedNTimes([5, 1, 5, 2, 5, 3, 5, 4]))

    def test_977_sortedSquares(self):
        self.assertEqual([0, 1, 9, 16, 100], self.solution._977_sortedSquares([-4, -1, 0, 3, 10]))

    def test_1002_commonChars(self):
        self.assertEqual(["e", "l", "l"], self.solution._1002_commonChars(["bella", "label", "roller"]))
        self.assertEqual(["c", "o"], self.solution._1002_commonChars(["cool", "lock", "cook"]))

    def test_1008_bstFromPreorder(self):
        expected = TreeNode(8)
        expected.left = TreeNode(5)
        expected.right = TreeNode(10)
        expected.left.left = TreeNode(1)
        expected.left.right = TreeNode(7)
        expected.right.right = TreeNode(12)

        actual = self.solution._1008_bstFromPreorder([8, 5, 1, 7, 10, 12])

        self.assertEqual(expected.val, actual.val)
        self.assertEqual(expected.left.val, actual.left.val)
        self.assertEqual(expected.right.val, actual.right.val)
        self.assertEqual(expected.left.left.val, actual.left.left.val)
        self.assertEqual(expected.left.right.val, actual.left.right.val)
        self.assertEqual(expected.right.right.val, actual.right.right.val)

    def test_1021_removeOuterParentheses(self):
        self.assertEqual("()()()()(())", self.solution._1021_removeOuterParentheses("(()())(())(()(()))"))
        self.assertEqual("()()()", self.solution._1021_removeOuterParentheses("(()())(())"))
        self.assertEqual("", self.solution._1021_removeOuterParentheses("()()"))

    def test_1046_lastStoneWeight(self):
        self.assertEqual(1, self.solution._1046_lastStoneWeight([2, 7, 4, 1, 8, 1]))

    def test_1047_removeDuplicates(self):
        self.assertEqual("ca", self.solution._1047_removeDuplicates("abbaca"))
        self.assertEqual("a", self.solution._1047_removeDuplicates("a"))
        self.assertEqual("", self.solution._1047_removeDuplicates(""))
        self.assertIsNone(self.solution._1047_removeDuplicates(None))

    def test_1051_heightChecker(self):
        self.assertEqual(3, self.solution._1051_heightChecker([1, 1, 4, 2, 1, 3]))

    def test_1078_findOcurrences(self):
        self.assertEqual(["girl", "student"],
                         self.solution._1078_findOcurrences("alice is a good girl she is a good student", "a", "good"))
        self.assertEqual(["we", "rock"], self.solution._1078_findOcurrences("we will we will rock you", "we", "will"))

    def test_1089_duplicateZeros(self):
        param = [1, 0, 2, 3, 0, 4, 5, 0]
        self.solution._1089_duplicateZeros(param)

        self.assertEqual([1, 0, 0, 2, 3, 0, 0, 4], param)

    def test_1108_defangIPaddr(self):
        self.assertEqual("192[.]168[.]1[.]1", self.solution._1108_defangIPaddr("192.168.1.1"))

    def test_1122_relativeSortArray(self):
        expected = [2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]
        self.assertEqual(expected,
                         self.solution._1122_relativeSortArray([2, 3, 1, 3, 2, 4, 6, 19, 9, 2, 7], [2, 1, 4, 3, 9, 6]))

    def test_1154_dayOfYear(self):
        self.assertEqual(9, self.solution._1154_dayOfYear("2019-01-09"))
        self.assertEqual(41, self.solution._1154_dayOfYear("2019-02-10"))
        self.assertEqual(60, self.solution._1154_dayOfYear("2003-03-01"))
        self.assertEqual(61, self.solution._1154_dayOfYear("2004-03-01"))

    def test_1160_countCharacters(self):
        self.assertEqual(6, self.solution._1160_countCharacters(["cat", "bt", "hat", "tree"], "attach"))
        self.assertEqual(10, self.solution._1160_countCharacters(["hello", "world", "leetcode"], "welldonehoneyr"))

        self.assertEqual(0, self.solution._1160_countCharacters([
            "dyiclysmffuhibgfvapygkorkqllqlvokosagyelotobicwcmebnpznjbirzrzsrtzjxhsfpiwyfhzyonmuabtlwin",
            "ndqeyhhcquplmznwslewjzuyfgklssvkqxmqjpwhrshycmvrb", "ulrrbpspyudncdlbkxkrqpivfftrggemkpyjl",
            "boygirdlggnh", "xmqohbyqwagkjzpyawsydmdaattthmuvjbzwpyopyafphx",
            "nulvimegcsiwvhwuiyednoxpugfeimnnyeoczuzxgxbqjvegcxeqnjbwnbvowastqhojepisusvsidhqmszbrnynkyop",
            "hiefuovybkpgzygprmndrkyspoiyapdwkxebgsmodhzpx",
            "juldqdzeskpffaoqcyyxiqqowsalqumddcufhouhrskozhlmobiwzxnhdkidr", "lnnvsdcrvzfmrvurucrzlfyigcycffpiuoo",
            "oxgaskztzroxuntiwlfyufddl",
            "tfspedteabxatkaypitjfkhkkigdwdkctqbczcugripkgcyfezpuklfqfcsccboarbfbjfrkxp",
            "qnagrpfzlyrouolqquytwnwnsqnmuzphne", "eeilfdaookieawrrbvtnqfzcricvhpiv",
            "sisvsjzyrbdsjcwwygdnxcjhzhsxhpceqz", "yhouqhjevqxtecomahbwoptzlkyvjexhzcbccusbjjdgcfzlkoqwiwue",
            "hwxxighzvceaplsycajkhynkhzkwkouszwaiuzqcleyflqrxgjsvlegvupzqijbornbfwpefhxekgpuvgiyeudhncv",
            "cpwcjwgbcquirnsazumgjjcltitmeyfaudbnbqhflvecjsupjmgwfbjo", "teyygdmmyadppuopvqdodaczob",
            "qaeowuwqsqffvibrtxnjnzvzuuonrkwpysyxvkijemmpdmtnqxwekbpfzs",
            "qqxpxpmemkldghbmbyxpkwgkaykaerhmwwjonrhcsubchs"],
            "usdruypficfbpfbivlrhutcgvyjenlxzeovdyjtgvvfdjzcmikjraspdfp"))

    def test_1360_daysBetweenDates(self):
        self.assertEqual(1, self.solution._1360_daysBetweenDates("2019-06-29", "2019-06-30"))
        self.assertEqual(15, self.solution._1360_daysBetweenDates("2020-01-15", "2019-12-31"))

    def test_1365_smallerNumbersThanCurrent(self):
        self.assertEqual([4, 0, 1, 1, 3], self.solution._1365_smallerNumbersThanCurrent([8, 1, 2, 2, 3]))
        self.assertEqual([2, 1, 0, 3], self.solution._1365_smallerNumbersThanCurrent([6, 5, 4, 8]))
        self.assertEqual([0, 0, 0, 0], self.solution._1365_smallerNumbersThanCurrent([7, 7, 7, 7]))

    def test_1374_generateTheString(self):
        self.assertEqual("aaab", self.solution._1374_generateTheString(4))
        self.assertEqual("ab", self.solution._1374_generateTheString(2))
        self.assertEqual("aaaaaaa", self.solution._1374_generateTheString(7))

    def test_1380_luckyNumbers(self):
        matrix = [[3, 7, 8], [9, 11, 13], [15, 16, 17]]

        self.assertEqual([15], self.solution._1380_luckyNumbers(matrix))

    def test_1389_createTargetArray(self):
        self.assertEqual([0, 4, 1, 3, 2], self.solution._1389_createTargetArray([0, 1, 2, 3, 4], [0, 1, 2, 2, 1]))
        self.assertEqual([0, 1, 2, 3, 4], self.solution._1389_createTargetArray([1, 2, 3, 4, 0], [0, 1, 2, 3, 0]))
        self.assertEqual([1], self.solution._1389_createTargetArray([1], [0]))

    def test_1394_findLucky(self):
        self.assertEqual(2, self.solution._1394_findLucky([2, 2, 3, 4]))
        self.assertEqual(3, self.solution._1394_findLucky([1, 2, 2, 3, 3, 3]))
        self.assertEqual(-1, self.solution._1394_findLucky([2, 2, 2, 3, 3]))
        self.assertEqual(-1, self.solution._1394_findLucky([5]))

    def test_1399_countLargestGroup(self):
        self.assertEqual(4, self.solution._1399_countLargestGroup(13))
        self.assertEqual(2, self.solution._1399_countLargestGroup(2))
        self.assertEqual(6, self.solution._1399_countLargestGroup(15))
        self.assertEqual(5, self.solution._1399_countLargestGroup(24))

    def test_1403_minSubsequence(self):
        self.assertEqual([10, 9], self.solution._1403_minSubsequence([4, 3, 10, 9, 8]))
        self.assertEqual([7, 7, 6], self.solution._1403_minSubsequence([4, 4, 7, 6, 7]))


if __name__ == '__main__':
    unittest.main()
