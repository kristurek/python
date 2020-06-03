from typing import List
from collections import deque
from common import ListNode
from common import TreeNode


class Solution2:

    def _0_simple(self) -> bool:
        return True

    def _1_twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}

        for i, num in enumerate(nums):
            if (target - num) in map:
                return [map.get(target - num), i]

            map[num] = i

        raise ValueError("Illegal argument exception")

    def _7_reverse(self, x: int) -> int:
        minus = True if x < 0 else False
        x = abs(x)
        number = 0

        while x > 0:
            number = number * 10 + x % 10
            x = int(x / 10)

        return number if not minus else number * -1

    def _9_isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False

        number = x
        reverse = 0

        while x > 0:
            reverse = reverse * 10 + x % 10
            x = int(x / 10)

        return number == reverse

    def _13_romanToInt(self, s: str) -> int:
        map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000, }

        sum = 0
        for i in range(0, len(s) - 1):
            cVal = map[s[i]]
            nVal = map[s[i + 1]]

            if cVal < nVal:
                sum -= cVal
            else:
                sum += cVal

        return sum + map[s[len(s) - 1]]

    def _14_longestCommonPrefix(self, strs: List[str]) -> str:
        longestPrefix = strs[0]

        for string in strs:
            while string.find(longestPrefix) != 0:
                longestPrefix = longestPrefix[0:len(longestPrefix) - 1]

        return longestPrefix

    def _20_isValid(self, s: str) -> bool:
        stack = []

        for cChar in s:
            if cChar == '(' or cChar == '{' or cChar == '[':
                stack.append(cChar)
            elif cChar == ')' and stack[-1] == '(':
                stack.pop()
            elif cChar == '}' and stack[-1] == '{':
                stack.pop()
            elif cChar == ']' and stack[-1] == '[':
                stack.pop()

        return True if len(stack) == 0 else False

    def _21_mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        l3 = ListNode(None)
        head = l3

        while l1 is not None and l2 is not None:
            if l1.val <= l2.val:
                l3.next = l1

                l1 = l1.next
                l3 = l3.next
            else:
                l3.next = l2

                l2 = l2.next
                l3 = l3.next

        if l1 is not None:
            l3.next = l1

        if l2 is not None:
            l3.next = l2

        return head.next

    def _26_removeDuplicates(self, nums: List[int]) -> int:
        slow = 0
        for fast in range(0, len(nums)):
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]

        return slow + 1

    def _27_removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        for fast in range(0, len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1

        return slow

    def _28_strStr(self, haystack: str, needle: str) -> int:
        j = 0
        i = 0

        while i < len(haystack):
            if haystack[i] == needle[j]:
                j += 1

                if j == len(needle):
                    return i - len(needle) + 1
            else:
                i = i - j if j > 0 else i
                j = 0
            i += 1

        return -1

    def _35_searchInsert(self, nums: List[int], target: int) -> int:
        l = 0
        h = len(nums) - 1

        while l <= h:
            m = l + int((h - l) / 2)

            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                h = m - 1
            else:
                return m

        return l

    def _53_maxSubArray(self, nums: List[int]) -> int:
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

    def _58_lengthOfLastWord(self, s: str) -> int:
        ss = s.strip().split(' ')
        return len(ss[len(ss) - 1])

    def _66_plusOne(self, digits: List[int]) -> List[int]:
        for i in reversed(range(len(digits))):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1
                return digits

        digits = [0] * (len(digits) + 1)
        digits[0] = 1

        return digits

    def _67_addBinary(self, a: str, b: str) -> str:
        i = len(a) - 1
        j = len(b) - 1
        carry = 0
        c = []

        while i >= 0 or j >= 0:
            sum = carry
            if i >= 0:
                sum += int(a[i])
                i -= 1
            if j >= 0:
                sum += int(b[j])
                j -= 1

            c.append(str(int(sum % 2)))
            carry = int(sum / 2)

        if carry != 0:
            c.append('1')

        return "".join(reversed(c))

    def _69_mySqrt(self, x: int) -> int:
        l = 0
        h = x
        answer = -1

        while l <= h:
            m = l + int((h - l) / 2)
            if m * m < x:
                l = m + 1
                answer = m
            elif m * m > x:
                h = m - 1
            else:
                return m

        return answer

    def _70_climbStairs(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2

        firstStep = 1
        secondStep = 2

        for i in range(2, n):
            thirdStep = firstStep + secondStep

            firstStep = secondStep
            secondStep = thirdStep

        return secondStep

    def _83_deleteDuplicates(self, ln: ListNode) -> ListNode:
        current = ln
        head = current

        while ln is not None:
            if ln.val != current.val:
                current.next = ln
                current = current.next

            ln = ln.next

        current.next = None

        return head

    def _88_merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i = m - 1
        j = n - 1
        k = m + n - 1

        while i >= 0 or j >= 0:
            if i >= 0 and j >= 0:
                if nums1[i] >= nums2[j]:
                    nums1[k] = nums1[i]
                    k -= 1
                    i -= 1
                else:
                    nums1[k] = nums2[j]
                    k -= 1
                    j -= 1
            elif i >= 0:
                nums1[k] = nums1[i]
                k -= 1
                i -= 1
            else:
                nums1[k] = nums2[j]
                k -= 1
                j -= 1

    def _100_isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        queue1 = deque()
        queue2 = deque()

        queue1.append(p)
        queue2.append(q)

        while len(queue1) != 0 and len(queue2) != 0:
            tn1 = queue1.popleft()
            tn2 = queue2.popleft()

            if tn1.left is None and tn2.left is not None:
                return False
            if tn1.right is None and tn2.right is not None:
                return False
            if tn2.left is None and tn1.left is not None:
                return False
            if tn2.right is None and tn1.right is not None:
                return False
            if tn1.val != tn2.val:
                return False

            if tn1.left is not None:
                queue1.append(tn1.left)
            if tn1.right is not None:
                queue1.append(tn1.right)

            if tn2.left is not None:
                queue2.append(tn2.left)
            if tn2.right is not None:
                queue2.append(tn2.right)

        return len(queue1) == 0 and len(queue2) == 0


def main():
    pass


if __name__ == "__main__":
    main()
