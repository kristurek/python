import unittest

from common import ListNode
from common import TreeNode
from solution import Solution


class SolutionTest(unittest.TestCase):
    solution = Solution()

    def test_0_sample(self):
        self.assertTrue(self.solution._0_simple())

    def test_1_twoSum(self):
        self.assertCountEqual([0, 1], self.solution._1_twoSum([2, 7, 11, 15], 9))
        with self.assertRaises(ValueError):
            self.solution._1_twoSum([2, 7, 11, 15], 8)

    def test_7_reverse(self):
        self.assertEqual(321, self.solution._7_reverse(123))
        self.assertEqual(-321, self.solution._7_reverse(-123))

    def test_9_isPalindrome(self):
        self.assertTrue(self.solution._9_isPalindrome(121))
        self.assertFalse(self.solution._9_isPalindrome(-121))
        self.assertFalse(self.solution._9_isPalindrome(123))

    def test_13_romanToInt(self):
        self.assertEqual(9, self.solution._13_romanToInt('IX'))
        self.assertEqual(4, self.solution._13_romanToInt('IV'))
        self.assertEqual(3, self.solution._13_romanToInt('III'))

    def test_14_longestCommonPrefix(self):
        self.assertEqual("fl", self.solution._14_longestCommonPrefix(["flower", "flow", "flight"]))
        self.assertEqual("", self.solution._14_longestCommonPrefix(["dog", "racecar", "car"]))

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

    def test_35_searchInsert(self):
        self.assertEqual(2, self.solution._35_searchInsert([2, 3, 5, 6, 7], 5))

    def test_53_maxSubArray(self):
        self.assertEqual(6, self.solution._53_maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))

    def test_58_lengthOfLastWord(self):
        self.assertEqual(5, self.solution._58_lengthOfLastWord('Hello World'))

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

    def test_88_merge(self):
        a = [1, 2, 3, 0, 0, 0]
        b = [1, 2, 3]
        self.solution._88_merge(a, 3, b, 3)
        self.assertEqual([1, 1, 2, 2, 3, 3], a)

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

    def test_118_generate(self):
        results = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

        self.assertEqual(results, self.solution._118_generate(5))

    def test_119_getRow(self):
        results = [1, 4, 6, 4, 1]

        self.assertEqual(results, self.solution._119_getRow(4))

    def test_121_maxProfit(self):
        self.assertEqual(5, self.solution._121_maxProfit([7, 1, 5, 3, 6, 4]))
        self.assertEqual(0, self.solution._121_maxProfit([7, 6, 4, 3, 1]))

    def test_maxProfit_122(self):
        self.assertEqual(7, self.solution._122_maxProfit([7, 1, 5, 3, 6, 4]))
        self.assertEqual(4, self.solution._122_maxProfit([1, 2, 3, 4, 5]))
        self.assertEqual(0, self.solution._122_maxProfit([7, 6, 4, 3, 1]))

    def test_125_isPalindrome(self):
        self.assertEqual(True, self.solution._125_isPalindrome("A man, a plan, a canal: Panama"))
        self.assertEqual(False, self.solution._125_isPalindrome("race a car"))

    def test_136_singleNumber(self):
        self.assertEqual(1, self.solution._136_singleNumber([2, 2, 1]))
        self.assertEqual(4, self.solution._136_singleNumber([4, 1, 2, 1, 2]))

    def test_141_hasCycle(self):
        head = ListNode(0)
        head.next = ListNode(1)
        head.next.next = ListNode(2)
        head.next.next.next = ListNode(3)
        head.next.next.next.next = head.next

        self.assertEqual(True, self.solution._141_hasCycle(head))

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

    def test_189_rotate(self):
        nums = [1, 2, 3, 4, 5, 6, 7]
        self.solution._189_rotate(nums, 3)
        self.assertEqual([5, 6, 7, 1, 2, 3, 4], nums)

    def test_198_rob(self):
        self.assertEqual(4, self.solution._198_rob([1, 2, 3, 1]))
        self.assertEqual(12, self.solution._198_rob([2, 7, 9, 3, 1]))

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

    def test_231_isPowerOfTwo(self):
        self.assertTrue(self.solution._231_isPowerOfTwo(1))
        self.assertTrue(self.solution._231_isPowerOfTwo(16))
        self.assertFalse(self.solution._231_isPowerOfTwo(218))

    def test_234_isPalindrome(self):
        head = ListNode(1)
        head.next = ListNode(2)
        self.assertFalse(self.solution._234_isPalindrome(head))

        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(2)
        head.next.next.next = ListNode(1)
        self.assertTrue(self.solution._234_isPalindrome(head))

    def test_283_moveZeroes(self):
        nums = [0, 1, 0, 3, 12]
        answer = [1, 3, 12, 0, 0]
        self.solution._283_moveZeroes(nums)

        self.assertEqual(answer, nums)


if __name__ == '__main__':
    unittest.main()
