from typing import List

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


def main():
    pass


if __name__ == "__main__":
    main()
