from typing import List

from common import ListNode


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


def main():
    pass


if __name__ == "__main__":
    main()
