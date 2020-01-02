from collections import deque
from typing import List

from common import ListNode
from common import TreeNode

import math

class Solution:

    def simple(self) -> bool:
        return True

    def two_sum_1(self, nums: List[int], target: int) -> List[int]:
        map = {}

        for i, num in enumerate(nums):
            if (target - num) in map:
                return [map.get(target - num), i]
            else:
                map[num] = i

        raise ValueError("No found")

    def reverse_7(self, x: int) -> int:
        minus = 1;
        if x < 0:
            minus = -1;
            x = x * -1

        number = x
        rNumber = 0;

        while number > 0:
            rNumber = rNumber * 10 + number % 10
            number = int(number / 10)

        return int(rNumber) * minus

    def isPalindrome_9(self, x: int) -> bool:
        if x < 0:
            return False
        if x == 0:
            return True

        number = x
        rNumber = 0;

        while number > 0:
            rNumber = rNumber * 10 + number % 10
            number = int(number / 10)

        return rNumber == x

    def romanToInt_13(self, s: str) -> int:
        dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000, }
        year = 0

        for i in range(len(s) - 1):
            currentIntValue = dict[s[i]]
            nextIntValue = dict[s[i + 1]]
            if currentIntValue >= nextIntValue:
                year += currentIntValue
            else:
                year -= currentIntValue

        return year + dict[s[len(s) - 1]]

    def longestCommonPrefix_14(self, strs: List[str]) -> str:
        strs = sorted(strs, key=len)

        prefix = strs[0]
        for str in strs:
            while str.find(prefix) != 0:
                prefix = prefix[0:len(prefix) - 1]

        return prefix

    def isValid_20(self, s: str) -> bool:
        stack = []

        for bracket in s:
            if bracket == '(' or bracket == '{' or bracket == '[':
                stack.append(bracket)
            else:
                if bracket == ')' and stack[-1] == '(':
                    stack.pop()
                elif bracket == '}' and stack[-1] == '{':
                    stack.pop()
                elif bracket == ']' and stack[-1] == '[':
                    stack.pop()
                else:
                    return False
        return stack == []

    def mergeTwoLists_21(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(None)
        current = head

        while l1 and l2:
            if l1.val <= l2.val:
                current.next = ListNode(l1.val)
                l1 = l1.next
            else:
                current.next = ListNode(l2.val)
                l2 = l2.next

            current = current.next

        while l1:
            current.next = ListNode(l1.val)
            l1 = l1.next
            current = current.next

        while l2:
            current.next = ListNode(l2.val)
            l2 = l2.next
            current = current.next

        return head.next

    def removeDuplicates_26(self, nums: List[int]) -> int:
        j = 0
        for i in range(0, len(nums) - 1):
            if nums[i] != nums[i + 1]:
                nums[j] = nums[i]
                j += 1
            i += 1

        nums[j] = nums[len(nums) - 1]
        return j + 1

    def removeElement_27(self, nums: List[int], val: int) -> int:
        j = 0
        for i in range(0, len(nums) - 1):
            if nums[i] != val:
                nums[j] = nums[i]
                j += 1
            i += 1

        nums[j] = nums[len(nums) - 1]
        return j + 1
        pass

    def strStr_28(self, haystack: str, needle: str) -> int:
        i = 0
        j = 0
        while i < len(haystack):
            if haystack[i] == needle[j]:
                j += 1
            else:
                if j != 0:
                    i = i - j
                j = 0
            i += 1

            if j == len(needle):
                return i - len(needle)

        return -1

    def searchInsert_35(self, nums: List[int], target: int) -> int:
        l = 0
        h = len(nums) - 1

        while l < h:
            mid = l + int((h - l) / 2)
            if nums[mid] < target:
                l = mid + 1
            elif nums[mid] > target:
                h = mid - 1
            else:
                return mid
        return l

    def maxSubArray_53(self, nums: List[int]) -> int:
        sum = 0
        max = float('-inf')

        for num in nums:
            if sum < 0:
                sum = num
            else:
                sum += num

            if sum > max:
                max = sum

        return max

    def lengthOfLastWord_58(self, s: str) -> int:
        s = str.strip(s)

        i = len(s) - 1
        size = i
        while i >= 0:
            if s[i] == ' ':
                return size - i
            i -= 1

        return 0

    def plusOne_66(_self, digits: List[int]) -> List[int]:
        i = len(digits) - 1

        while i >= 0:
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] = digits[i] + 1
                return digits
            i -= 1

        digits.insert(0, 1)

        return digits

    def addBinary_67(self, a: str, b: str) -> str:
        i = len(a) - 1
        j = len(b) - 1

        carry = 0
        digits = []

        while i >= 0 or j >= 0:
            sum = carry
            if i >= 0:
                sum += int(a[i])
            if j >= 0:
                sum += int(b[j])

            digits.append(sum % 2)
            carry = int(sum / 2)

            i -= 1
            j -= 1

        if carry != 0:
            digits.append(1)

        return ''.join(map(str, reversed(digits)))

    def mySqrt_69(self, x: int) -> int:
        l = 0
        h = x
        ans = -1

        while (l <= h):
            m = l + int((h - l) / 2)

            if m * m < x:
                l = m + 1
                ans = m
            elif m * m > x:
                h = m - 1
            else:
                return m

        return ans

    def climbStairs_70(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2

        firstStep = 1
        secondStep = 2

        for i in range(3, n + 1):
            thirdStep = firstStep + secondStep

            firstStep = secondStep
            secondStep = thirdStep

        return secondStep

    def deleteDuplicates_83(self, head: ListNode) -> ListNode:
        currentNode = head

        while currentNode and currentNode.next:
            if currentNode.val == currentNode.next.val:
                currentNode.next = currentNode.next.next
            else:
                currentNode = currentNode.next

        return head

    def merge_88(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        m = m - 1
        n = n - 1
        k = m + n + 1

        while k >= 0:
            if m >= 0 and n < 0:
                nums1[k] = nums1[m]
                k -= 1
                m -= 1
            elif m < 0 and n >= 0:
                nums1[k] = nums2[n]
                k -= 1
                n -= 1
            else:
                if nums1[m] > nums2[n]:
                    nums1[k] = nums1[m]
                    k -= 1
                    m -= 1
                elif nums1[m] < nums2[n]:
                    nums1[k] = nums2[n]
                    k -= 1
                    n -= 1
                else:
                    nums1[k] = nums1[m]
                    k -= 1
                    m -= 1
                    nums1[k] = nums2[n]
                    k -= 1
                    n -= 1
        pass

    def isSameTree_100(self, p: TreeNode, q: TreeNode) -> bool:
        queue1 = deque()
        queue2 = deque()

        queue1.append(p)
        queue2.append(q)

        while len(queue1) > 0 and len(queue2) > 0:
            tn1 = queue1.popleft()
            tn2 = queue2.popleft()

            if tn1.val != tn2.val:
                return False
            elif tn1.left == None and tn2.left != None:
                return False
            elif tn1.left != None and tn2.left == None:
                return False
            elif tn1.right == None and tn2.right != None:
                return False
            elif tn1.right != None and tn2.right == None:
                return False

            if tn1.left != None:
                queue1.append(tn1.left)
            if tn1.right != None:
                queue1.append(tn1.right)

            if tn2.left != None:
                queue2.append(tn2.left)
            if tn2.right != None:
                queue2.append(tn2.right)

        return len(queue1) == 0 and len(queue2) == 0

    def isSymmetric_101(self, root: TreeNode) -> bool:
        queue = deque()

        queue.append(root)
        queue.append(root)

        while len(queue) > 1:
            tn1 = queue.popleft()
            tn2 = queue.popleft()

            if tn1 == None and tn2 == None:
                continue
            elif tn1 != None and tn2 == None:
                return False
            elif tn1 == None and tn2 != None:
                return False
            elif tn1.val != tn2.val:
                return False
            else:
                queue.append(tn1.left)
                queue.append(tn2.right)
                queue.append(tn1.right)
                queue.append(tn2.left)

        return True

    def maxDepth_104(self, root: TreeNode) -> int:
        queue = deque()
        queue.append(root)
        depth = 0

        while len(queue) > 0:
            size = len(queue)
            while size > 0:
                size -= 1

                tn = queue.popleft()
                if tn.left is not None:
                    queue.append(tn.left)
                if tn.right is not None:
                    queue.append(tn.right)

            depth += 1

        return depth

    def levelOrderBottom_107(self, root: TreeNode) -> List[List[int]]:
        queue = deque()
        queue.append(root)
        levels = []

        while len(queue) > 0:
            size = len(queue)
            level = []
            while size > 0:
                size -= 1

                tn = queue.popleft()
                level.append(tn.val)

                if tn.left is not None:
                    queue.append(tn.left)
                if tn.right is not None:
                    queue.append(tn.right)

            levels.append(level)

        levels.reverse()

        return levels

    def sortedArrayToBST_108(self, nums: List[int]) -> TreeNode:
        return self.sortedArrayToBST_108_recursive(nums, 0, len(nums) - 1)

    def sortedArrayToBST_108_recursive(self, nums: List[int], l: int, h: int) -> TreeNode:
        if l > h:
            return None

        m = math.ceil(l + (h - l) / 2)
        node = TreeNode(nums[m])
        node.left = self.sortedArrayToBST_108_recursive(nums, l, m - 1)
        node.right = self.sortedArrayToBST_108_recursive(nums, m + 1, h)

        return node


def main():
    pass


if __name__ == "__main__":
    main()
