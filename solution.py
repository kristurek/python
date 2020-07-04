import sys
from collections import deque, OrderedDict
from functools import cmp_to_key
from string import ascii_lowercase
from typing import List, Dict

from common import Employee, KeyPriorityQueue
from common import ListNode
from common import Node
from common import TreeNode


class MinStack:

    def __init__(self):
        self.stack = deque()
        self.min = sys.maxsize

    def push(self, x: int) -> None:
        if x <= self.min:
            self.stack.append(self.min)
            self.min = x
        self.stack.append(x)

    def pop(self) -> None:
        x = self.stack.pop()
        if x == self.min:
            self.min = self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min


class HashEntry:
    def __init__(self, value: int):
        self.value = value
        self.next = None


class MyHashSet:

    def __init__(self):
        self.SIZE = 128
        self.buckets = [None] * self.SIZE

    def add(self, value: int) -> None:
        hash = self.generateHash(value)
        he = self.buckets[hash]

        if he == None:
            self.buckets[hash] = HashEntry(value)
        else:
            while he != None:
                if he.value == value:
                    he.value = value
                    break
                else:
                    if he.next == None:
                        he.next = HashEntry(value)
                        break
                    else:
                        he = he.next

    def remove(self, value: int) -> None:
        hash = self.generateHash(value)
        cHe = self.buckets[hash]
        pHe = None

        while cHe != None:
            if cHe.value != value:
                pHe = cHe
                cHe = cHe.next
            else:
                if pHe == None:  # Remove first
                    self.buckets[hash] = cHe.next
                elif cHe.next == None:  # Remove last
                    pHe.next = None
                else:  # Remove middle
                    pHe.next = cHe.next
                break

    def contains(self, value: int) -> bool:
        hash = self.generateHash(value)
        eb = self.buckets[hash]

        while eb:
            if eb.value == value:
                return True
            eb = eb.next

        return False

    def generateHash(self, value: int) -> int:
        return value * 31 % self.SIZE


class Solution:

    def _0_simple(self) -> bool:
        return True

    def _1_twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}

        for i, num in enumerate(nums):
            if (target - num) in map:
                return [map.get(target - num), i]

            map[num] = i

        raise ValueError("Illegal argument exception")

    def _2_addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        l3 = ListNode(-1)
        head = l3
        carry = 0

        while l1 or l2:
            sum = (l1.val if l1 != None else 0) + (l2.val if l2 != None else 0) + carry
            carry = int(sum / 10)
            l3.next = ListNode(sum % 10)
            l3 = l3.next

            if l1 != None:
                l1 = l1.next

            if l2 != None:
                l2 = l2.next

        if carry != 0:
            l3.next = ListNode(carry)

        return head.next

    def _3_lengthOfLongestSubstring(self, s: str) -> int:
        queue = deque()
        maxLength = 0

        for c in s:
            if c not in queue:
                queue.append(c)
                maxLength = max(maxLength, len(queue))
            else:
                while c in queue:
                    queue.popleft()
                queue.append(c)

        return maxLength

    def _5_longestPalindrome(self, s: str) -> str:
        if s == None or len(s) <= 1:
            return s

        longestPalindrome = ""

        for i in range(len(s)):
            tmp = self._5_longestPalindrome_findPalindromeByExtend(s, i, i)  # odd length of palindrome
            longestPalindrome = longestPalindrome if len(longestPalindrome) > len(tmp) else tmp

            tmp = self._5_longestPalindrome_findPalindromeByExtend(s, i, i + 1)  # even length of palindrome
            longestPalindrome = longestPalindrome if len(longestPalindrome) > len(tmp) else tmp

        return longestPalindrome

    def _5_longestPalindrome_findPalindromeByExtend(self, s: str, begin: int, end: int) -> str:
        while begin >= 0 and end < len(s) and s[begin] == s[end]:
            begin -= 1
            end += 1

        return s[begin + 1:end]

    def _7_reverse(self, x: int) -> int:
        min_limit = -0x80000000  # hex(-2**31-1) Integer.MIN_VALUE
        max_limit = 0x7fffffff  # hex(2**31-1) Integer.MAX_VALUE

        minus = True if x < 0 else False
        x = abs(x)
        number = 0

        while x > 0:
            number = number * 10 + x % 10
            x = int(x / 10)

        if number & max_limit == number:
            return number * -1 if minus else number
        return 0

    def _9_isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False

        number = x
        reverse = 0

        while x > 0:
            reverse = reverse * 10 + x % 10
            x = int(x / 10)

        return number == reverse

    def _11_maxArea(self, height: List[int]) -> int:
        l = 0
        h = len(height) - 1
        maxArea = 0

        while l < h:
            localArea = min(height[l], height[h]) * (h - l)
            maxArea = max(maxArea, localArea)

            if height[l] <= height[h]:
                l += 1
            else:
                h -= 1

        return maxArea

    def _12_intToRoman(self, num: int) -> str:
        map = OrderedDict()
        map[1000] = "M"
        map[900] = "CM"
        map[500] = "D"
        map[400] = "CD"
        map[100] = "C"
        map[90] = "XC"
        map[50] = "L"
        map[40] = "XL"
        map[10] = "X"
        map[9] = "IX"
        map[5] = "V"
        map[4] = "IV"
        map[1] = "I"

        roman = ""

        for item in map.items():
            while num >= item[0]:
                num = num - item[0]
                roman += item[1]

        return roman

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
        if strs == None or len(strs) == 0:
            return ""

        longestPrefix = strs[0]

        for string in strs:
            while string.find(longestPrefix) != 0:
                longestPrefix = longestPrefix[0:len(longestPrefix) - 1]

        return longestPrefix

    def _15_threeSum(self, nums: List[int]) -> List[List[int]]:
        groups = set()
        nums = sorted(nums)

        for i in range(len(nums)):
            l = i + 1
            h = len(nums) - 1

            while l < h:
                sum = nums[i] + nums[l] + nums[h]
                if sum < 0:
                    l += 1
                elif sum > 0:
                    h -= 1
                else:
                    groups.add(tuple([nums[i], nums[l], nums[h]]))
                    l += 1
                    h -= 1

        return [list(t) for t in groups]

    def _16_threeSumClosest(self, nums: List[int], target: int) -> int:
        nums = sorted(nums)

        min_limit = -0x80000000  # hex(-2**31-1) Integer.MIN_VALUE
        max_limit = 0x7fffffff  # hex(2**31-1) Integer.MAX_VALUE

        closestSum = max_limit
        minDiff = max_limit

        for i in range(len(nums)):
            l = i + 1
            h = len(nums) - 1

            while l < h:
                sum = nums[i] + nums[l] + nums[h]

                diff = min(minDiff, abs(target - sum))
                if diff < minDiff:
                    minDiff = diff
                    closestSum = sum

                if sum > target:
                    h -= 1
                elif sum < target:
                    l += 1
                else:
                    return sum

        return closestSum

    def _17_letterCombinations(self, digits: str) -> List[str]:
        map = {}
        map['1'] = []
        map['2'] = ['a', 'b', 'c']
        map['3'] = ['d', 'e', 'f']
        map['4'] = ['g', 'h', 'i']
        map['5'] = ['j', 'k', 'l']
        map['6'] = ['m', 'n', 'o']
        map['7'] = ['p', 'q', 'r', 's']
        map['8'] = ['t', 'u', 'v']
        map['9'] = ['w', 'x', 'y', 'z']
        map['0'] = []

        groups = []

        if len(digits) > 0:
            self._17_letterCombinations_backtracking(groups, [], digits, 0, map)

        return groups

    def _17_letterCombinations_backtracking(self, output: list, tmp: list, digits: str, digitIndex: int,
                                            map: dict) -> None:
        if len(tmp) == len(digits):
            output.append("".join(tmp))
        else:
            chars = map[digits[digitIndex]]
            for cChar in chars:
                tmp.append(cChar)
                self._17_letterCombinations_backtracking(output, tmp, digits, digitIndex + 1, map)
                tmp.pop()

    def _18_fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        output = set()

        self._18_fourSum_backtracking(output, [], nums, target, 0)

        return [list(t) for t in output]

    def _18_fourSum_backtracking(self, output: set, tmp: List[int], nums: List[int], target: int,
                                 begin: int) -> None:
        if len(tmp) == 4:
            if sum(tmp) == target:
                output.add(tuple(sorted(tmp)))
        else:
            for i in range(begin, len(nums)):
                tmp.append(nums[i])
                self._18_fourSum_backtracking(output, tmp, nums, target, i + 1)
                tmp.pop()

    def _19_removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        nodes = []
        current = head

        while current:
            nodes.append(current)
            current = current.next

        if len(nodes) - n == 0:
            return head.next
        else:
            ln = nodes[len(nodes) - n - 1]
            ln.next = ln.next.next
            return head

    def _20_isValid(self, s: str) -> bool:
        if str == None:
            return False
        if len(s) == 0:
            return True

        stack = []

        for cChar in s:
            if cChar == '(' or cChar == '{' or cChar == '[':
                stack.append(cChar)
            elif cChar == ')' and stack and stack[-1] == '(':
                stack.pop()
            elif cChar == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif cChar == ']' and stack and stack[-1] == '[':
                stack.pop()
            else:
                return False  # when input == ) or ] or }

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

    def _22_generateParenthesis(self, n: int) -> List[str]:
        output = []

        self._22_generateParenthesis_backtracking(output, "", n, 0, 0)

        return output

    def _22_generateParenthesis_backtracking(self, output: List[str], tmp: str, n: int, open: int,
                                             close: int) -> None:
        if len(tmp) == n * 2:
            output.append(tmp)
        else:
            if open >= close:
                if open < n:
                    self._22_generateParenthesis_backtracking(output, tmp + "(", n, open + 1, close)
                if close < n:
                    self._22_generateParenthesis_backtracking(output, tmp + ")", n, open, close + 1)

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
        if len(haystack) == 0 and len(needle) == 0:
            return 0

        if len(haystack) != 0 and len(needle) == 0:
            return 0

        if len(haystack) == 0 and len(needle) != 0:
            return -1

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

    def _33_search(self, nums: List[int], target: int) -> int:
        if nums == None or len(nums) == 0:
            return -1

        l = 0
        h = len(nums) - 1

        while l < h:
            m = l + int((h - l) / 2)
            if nums[m] > nums[h]:
                l = m + 1
            else:
                h = m

        p = l
        l = 0
        h = len(nums) - 1

        if target >= nums[p] and target <= nums[h]:
            l = p
        else:
            h = p - 1

        while l <= h:
            m = l + int((h - l) / 2)
            if nums[m] > target:
                h = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m

        return -1

    def _34_searchRange(self, nums: List[int], target: int) -> List[int]:
        l = self._34_searchRange_search(nums, target, True)
        r = self._34_searchRange_search(nums, target, False)

        return [l, r]

    def _34_searchRange_search(self, nums: List[int], target: int, left: bool) -> int:
        l = 0
        h = len(nums) - 1

        while l <= h:
            m = l + int((h - l) / 2)

            if target > nums[m]:
                l = m + 1
            elif target < nums[m]:
                h = m - 1
            else:
                if left:
                    if m > 0 and nums[m - 1] == target:
                        h = m - 1
                    else:
                        return m
                else:
                    if m < len(nums) - 1 and nums[m + 1] == target:
                        l = m + 1
                    else:
                        return m

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

    def _36_isValidSudoku(self, board: List[List[str]]) -> bool:
        hashSet = set()

        for row in range(len(board)):
            for col in range(len(board[row])):
                val = board[row][col]
                if val != ".":
                    r = "r" + str(row) + val
                    c = "c" + str(col) + val
                    b = "b" + str(int(row / 3)) + "-" + str(int(col / 3)) + val
                    if r in hashSet or c in hashSet or b in hashSet:
                        return False
                    else:
                        hashSet.add(r)
                        hashSet.add(c)
                        hashSet.add(b)
        return True

    def _39_combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        output = []

        self._39_combinationSum_backtracking(output, [], nums, target, 0)

        return output

    def _39_combinationSum_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int],
                                        remain: int, begin: int) -> None:
        if remain == 0:
            output.append(list(tmp))
        elif remain < 0:
            return
        else:
            for i in range(begin, len(nums)):
                tmp.append(nums[i])
                self._39_combinationSum_backtracking(output, tmp, nums, remain - nums[i], i)
                tmp.pop()

    def _40_combinationSum2(self, nums: List[int], target: int) -> List[List[int]]:
        output = []
        used = [False for i in range(len(nums))]

        nums = sorted(nums)

        self._40_combinationSum2_backtracking(output, [], nums, target, used, 0)

        return output

    def _40_combinationSum2_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int],
                                         remained: int, used: List[bool], begin: int) -> None:
        if remained < 0:
            return
        elif remained == 0:
            output.append(list(tmp))
        else:
            for i in range(begin, len(nums)):
                if used[i] or (i > begin and nums[i - 1] == nums[i] and used[i] == False):
                    continue

                tmp.append(nums[i])
                used[i] = True
                self._40_combinationSum2_backtracking(output, tmp, nums, remained - nums[i], used, i + 1)
                used[i] = False
                tmp.pop()

    def _43_multiply(self, num1: str, num2: str) -> str:
        sum = [0 for i in range(len(num1) + len(num2))]

        for i in reversed(range(0, len(num1))):
            carry = 0
            for j in reversed(range(0, len(num2))):
                tmp = sum[i + j + 1] + int(num1[i]) * int(num2[j]) + carry
                sum[i + j + 1] = tmp % 10
                carry = int(tmp / 10)
            sum[i] += carry

        firstNotZero = next((i for i, x in enumerate(sum) if x), None)
        if firstNotZero != None:
            return "".join((str(x) for x in sum[firstNotZero: len(sum)]))

        return "0"

    def _46_permute(self, nums: List[int]) -> List[List[int]]:
        output = []

        self._46_permute_backtracking(output, [], nums)

        return output

    def _46_permute_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int]) -> None:
        if len(tmp) == len(nums):
            output.append(list(tmp))
        else:
            for i in range(0, len(nums)):
                if nums[i] in tmp:
                    continue

                tmp.append(nums[i])
                self._46_permute_backtracking(output, tmp, nums)
                tmp.pop()

    def _47_permuteUnique(self, nums: List[int]) -> List[List[int]]:
        output = []
        used = [False for i in range(len(nums))]

        nums = sorted(nums)

        self._47_permuteUnique_backtracking(output, [], nums, used)

        return output

    def _47_permuteUnique_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int],
                                       used: List[bool]) -> None:
        if len(tmp) == len(nums):
            output.append(list(tmp))
        else:
            for i in range(0, len(nums)):
                if used[i] or (i > 0 and nums[i - 1] == nums[i] and used[i - 1] == False):
                    continue

                used[i] = True
                tmp.append(nums[i])
                self._47_permuteUnique_backtracking(output, tmp, nums, used)
                tmp.pop()
                used[i] = False

    def _49_groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashMap = {}

        for str in strs:
            sortedStr = "".join(sorted(str))
            listStr = hashMap.get(sortedStr, [])
            listStr.append(str)
            hashMap[sortedStr] = listStr

        return list(hashMap.values())

    def _50_myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1

        tmp = self._50_myPow(x, int(n / 2))

        if n % 2 == 0:
            return tmp * tmp
        else:
            if n > 0:
                return tmp * tmp * x
            else:
                return (tmp * tmp) / x

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

    def _54_spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if matrix == None or len(matrix) == 0:
            return []

        values = []

        beginCol = 0
        endCol = len(matrix[0]) - 1
        beginRow = 0
        endRow = len(matrix) - 1

        while beginCol <= endCol and beginRow <= endRow:
            for col in range(beginCol, endCol + 1):
                values.append(matrix[beginRow][col])
            beginRow += 1

            for row in range(beginRow, endRow + 1):
                values.append(matrix[row][endCol])
            endCol -= 1

            if beginRow <= endRow:
                for col in reversed(range(beginCol, endCol + 1)):
                    values.append(matrix[endRow][col])
            endRow -= 1

            if beginCol <= endCol:
                for row in reversed(range(beginRow, endRow + 1)):
                    values.append(matrix[row][beginCol])
            beginCol += 1

        return values

    def _58_lengthOfLastWord(self, s: str) -> int:
        ss = s.strip().split(' ')
        return len(ss[len(ss) - 1])

    def _60_getPermutation(self, n: int, k: int) -> str:
        nums = [x for x in range(1, n + 1)]
        output = []

        self._60_getPermutation_backtracking(output, [], nums)

        return "".join([str(x) for x in output[k - 1]])

    def _60_getPermutation_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int]) -> None:
        if len(tmp) == len(nums):
            output.append(list(tmp))
        else:
            for i in range(0, len(nums)):
                if nums[i] in tmp:
                    continue

                tmp.append(nums[i])
                self._60_getPermutation_backtracking(output, tmp, nums)
                tmp.pop()

    def _61_rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head == None:
            return None

        cNode = head
        length = 1
        while cNode.next != None:
            cNode = cNode.next
            length += 1

        cNode.next = head  # loop

        k = k % length
        for i in range(0, length - k):
            cNode = cNode.next

        head = cNode.next
        cNode.next = None  # cut loop

        return head

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

    def _71_simplifyPath(self, path: str) -> str:
        stack = deque()

        for dir in path.split("/"):
            if dir == "..":
                if stack:
                    stack.pop()
            elif dir != "" and dir != ".":
                stack.append(dir)

        return "/" + "/".join(stack) if stack else "/"

    def _74_searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if matrix == None or len(matrix) == 0:
            return False

        rLength = len(matrix)
        cLength = len(matrix[0])
        l = 0
        h = rLength * cLength - 1

        while l <= h:
            m = l + int((h - l) / 2)

            if target > matrix[int(m / cLength)][m % cLength]:
                l = m + 1
            elif target < matrix[int(m / cLength)][m % cLength]:
                h = m - 1
            else:
                return True

        return False

    def _75_sortColors(self, nums: List[int]) -> None:
        self._75_sortColors_quicksort(nums, 0, len(nums) - 1)
        pass

    def _75_sortColors_quicksort(self, nums: List[int], l: int, h: int) -> None:
        if l < h:
            splitPoint = self._75_sortColors_partitions(nums, l, h)

            self._75_sortColors_quicksort(nums, l, splitPoint - 1)
            self._75_sortColors_quicksort(nums, splitPoint + 1, h)

    def _75_sortColors_partitions(self, nums: List[int], l: int, h: int) -> int:
        pivot = nums[h]
        lowersIndex = l - 1

        for currentIndex in range(l, h):
            if nums[currentIndex] < pivot:
                lowersIndex += 1

                tmp = nums[lowersIndex]
                nums[lowersIndex] = nums[currentIndex]
                nums[currentIndex] = tmp

        lowersIndex += 1

        tmp = nums[lowersIndex]
        nums[lowersIndex] = nums[h]
        nums[h] = tmp

        return lowersIndex

    def _77_combine(self, n: int, k: int) -> List[List[int]]:
        output = []
        nums = range(1, n + 1)

        self._77_combine_backtracking(output, [], nums, k, 0)

        return output

    def _77_combine_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int], k: int,
                                 begin: int) -> None:
        if len(tmp) == k:
            output.append(list(tmp))
        else:
            for i in range(begin, len(nums)):
                tmp.append(nums[i])
                self._77_combine_backtracking(output, tmp, nums, k, i + 1)
                tmp.pop()

    def _78_subsets(self, nums: List[int]) -> List[List[int]]:
        output = []

        self._78_subsets_backtracking(output, [], nums, 0)

        return output

    def _78_subsets_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int], begin: int) -> None:
        output.append(list(tmp))
        for i in range(begin, len(nums)):
            tmp.append(nums[i])
            self._78_subsets_backtracking(output, tmp, nums, i + 1)
            tmp.pop()

    def _80_removeDuplicates(self, nums: List[int]) -> int:
        slow = 1
        count = 1

        for fast in range(1, len(nums)):
            if nums[fast] == nums[fast - 1]:
                count += 1
            else:
                count = 1

            if count <= 2:
                nums[slow] = nums[fast]
                slow += 1

        return slow

    def _81_search(self, nums: List[int], target: int) -> bool:
        l = 0
        h = len(nums) - 1

        while l <= h:
            m = l + int((h - l) / 2)

            if nums[m] == target:
                return True
            elif nums[m] < nums[h]:  # range m,h is sorted
                if target > nums[m] and target <= nums[h]:  # check sorted right side
                    l = m + 1
                else:
                    h = m - 1
            elif nums[m] > nums[h]:  # range l,m is sorted and range m,h unsorted
                if target >= nums[l] and target < nums[m]:  # check sorted left side
                    h = m - 1
                else:
                    l = m + 1
            elif nums[m] == nums[h]:  # duplicates
                h -= 1
            elif nums[l] == nums[m]:  # duplicates
                l += 1

        return False

    def _82_deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        slow = dummy
        fast = dummy.next

        while fast is not None:
            while fast.next is not None and fast.val == fast.next.val:
                fast = fast.next

            if slow.next != fast:  # duplicates detected
                slow.next = fast.next
                fast = fast.next
            else:  # no duplicates
                slow = slow.next
                fast = fast.next

        return dummy.next

    def _83_deleteDuplicates(self, ln: ListNode) -> ListNode:
        if ln is None:
            return None

        current = ln
        head = current

        while ln is not None:
            if ln.val != current.val:
                current.next = ln
                current = current.next

            ln = ln.next

        current.next = None

        return head

    def _86_partition(self, head: ListNode, x: int) -> ListNode:
        hBeforeX = ListNode(-1)
        cBeforeX = hBeforeX
        hAfterX = ListNode(-1)
        cAfterX = hAfterX

        while head != None:
            if head.val < x:
                cBeforeX.next = head
                cBeforeX = cBeforeX.next
            else:
                cAfterX.next = head
                cAfterX = cAfterX.next
            head = head.next

        cBeforeX.next = hAfterX.next
        cAfterX.next = None

        return hBeforeX.next

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

    def _90_subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        output = []

        nums = sorted(nums)

        self._90_subsetsWithDup_backtracking(output, [], nums, 0)

        return output

    def _90_subsetsWithDup_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int],
                                        begin: int) -> None:
        output.append(list(tmp))

        for i in range(begin, len(nums)):
            if i > begin and nums[i - 1] == nums[i]:
                continue

            tmp.append(nums[i])
            self._90_subsetsWithDup_backtracking(output, tmp, nums, i + 1)
            tmp.pop()

    def _92_reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        dummyHead = ListNode(-1)
        dummyHead.next = head
        cNode = dummyHead

        # Go to position m-1
        for i in range(1, m):
            if (cNode.next != None):
                cNode = cNode.next

        # reverse from m to n
        prevNode = None
        currNode = cNode.next
        nextNode = None
        tailNode = cNode.next

        for i in range(m, n + 1):
            nextNode = currNode.next

            currNode.next = prevNode

            prevNode = currNode
            currNode = nextNode

        # connect list (0 to m-1) with list (m to n)
        cNode.next = prevNode
        # connect list (m to n) with list (n+1, ...)
        tailNode.next = nextNode

        return dummyHead.next

    def _94_inorderTraversal(self, root: TreeNode) -> List[int]:
        stack = deque()
        values = []
        current = root

        while len(stack) > 0 or current != None:
            if current != None:
                stack.append(current)

                current = current.left
            else:
                current = stack.pop()

                values.append(current.val)

                current = current.right

        return values

    def _98_isValidBST(self, root: TreeNode) -> bool:
        stack = deque()
        current = root
        previous = None

        while stack or current != None:
            if current != None:
                stack.append(current)
                current = current.left
            else:
                current = stack.pop()

                if previous != None and current.val <= previous.val:
                    return False

                previous = current
                current = current.right

        return True

    def _100_isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False

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

    def _101_isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return True

        queue = deque()
        queue.append(root)
        queue.append(root)

        while len(queue) > 0:
            tn1 = queue.popleft()
            tn2 = queue.popleft()

            if tn1 == None and tn2 == None:
                continue
            if tn1 == None or tn2 == None:
                return False
            if tn1.val != tn2.val:
                return False

            queue.append(tn1.left)
            queue.append(tn2.right)
            queue.append(tn1.right)
            queue.append(tn2.left)

        return True

    def _104_maxDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0

        queue = deque()
        queue.append(root)
        level = 0

        while len(queue) > 0:
            level += 1
            size = len(queue)
            while size > 0:
                size -= 1

                tn = queue.popleft()

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

        return level

    def _107_levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []

        queue = deque()
        levels = deque()

        queue.append(root)

        while len(queue) > 0:
            size = len(queue)
            level = []
            while size > 0:
                size -= 1

                tn = queue.popleft()
                level.append(tn.val)

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

            levels.appendleft(level)

        return list(levels)

    def _108_sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        return self._108_sortedArrayToBST_recursive(nums, 0, len(nums) - 1)

    def _108_sortedArrayToBST_recursive(self, nums, l, h):
        if l <= h:
            m = l + int((h - l) / 2)
            tn = TreeNode(nums[m])

            tn.left = self._108_sortedArrayToBST_recursive(nums, l, m - 1)
            tn.right = self._108_sortedArrayToBST_recursive(nums, m + 1, h)
            return tn
        else:
            return None

    def _110_isBalanced(self, root: TreeNode) -> bool:
        if root is None:
            return True

        l = self._110_isBalanced_max_height(root.left)
        r = self._110_isBalanced_max_height(root.right)

        return abs(l - r) <= 1 and self._110_isBalanced(root.left) and self._110_isBalanced(root.right)

    def _110_isBalanced_max_height(self, root):  # search max in current subtree
        if root is None:
            return 0

        return 1 + max(self._110_isBalanced_max_height(root.left), self._110_isBalanced_max_height(root.right))

    def _111_minDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0

        level = 0
        queue = deque()
        queue.append(root)

        while len(queue) > 0:
            level += 1
            size = len(queue)
            while size > 0:
                size -= 1

                tn = queue.popleft()

                if tn.left == None and tn.right == None:
                    return level

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

        return level

    def _112_hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if root is None:
            return False

        stack = deque()
        sums = deque()

        stack.append(root)
        sums.append(root.val)

        while len(stack) > 0:
            tn = stack.pop()
            path = sums.pop()

            if tn.left == None and tn.right == None and sum == path:
                return True

            if tn.right != None:
                stack.append(tn.right)
                sums.append(tn.right.val + path)

            if tn.left != None:
                stack.append(tn.left)
                sums.append(tn.left.val + path)

        return False

    def _118_generate(self, numRows: int) -> List[List[int]]:
        if numRows <= 0:
            return []

        if numRows == 1:
            return [[1]]

        if numRows == 2:
            return [[1], [1, 1]]

        rows = []
        rows.append([1])
        rows.append([1, 1])

        for i in range(2, numRows):
            row = []
            for j in range(0, i + 1):
                if j == 0:
                    row.append(1)
                elif j == i:
                    row.append(1)
                else:
                    row.append(rows[i - 1][j - 1] + rows[i - 1][j])
            rows.append(row)

        return rows

    def _119_getRow(self, rowIndex: int) -> List[int]:
        if rowIndex < 0:
            return []

        if rowIndex == 0:
            return [1]

        if rowIndex == 1:
            return [1, 1]

        rows = []
        rows.append([1])
        rows.append([1, 1])

        for i in range(2, rowIndex + 1):
            row = []
            for j in range(0, i + 1):
                if j == 0:
                    row.append(1)
                elif j == i:
                    row.append(1)
                else:
                    row.append(rows[i - 1][j - 1] + rows[i - 1][j])
            rows.append(row)

        return rows[rowIndex]

    def _121_maxProfit(self, prices: List[int]) -> int:
        if prices is None or len(prices) == 0:
            return 0

        min = prices[0]
        max = 0

        for i in range(1, len(prices)):
            if prices[i] < min:
                min = prices[i]
            elif prices[i] - min > max:
                max = prices[i] - min

        return max

    def _122_maxProfit(self, prices: List[int]) -> int:
        max = 0

        for i in range(1, len(prices)):
            if prices[i] - prices[i - 1] > 0:
                max += prices[i] - prices[i - 1]

        return max

    def _125_isPalindrome(self, s: str) -> bool:
        if s is None:
            return False

        l = 0
        h = len(s) - 1

        while l <= h:
            if s[l].isalnum() == False:
                l += 1
                continue
            if s[h].isalnum() == False:
                h -= 1
                continue

            if s[l].lower() != s[h].lower():
                return False

            l += 1
            h -= 1

        return True

    def _136_singleNumber(self, nums: List[int]) -> int:
        values = set()

        for num in nums:
            if num not in values:
                values.add(num)
            else:
                values.remove(num)

        return values.pop()

    def _141_hasCycle(self, head: ListNode) -> bool:
        duplicates = set()

        while head != None:
            if head in duplicates:
                return True
            else:
                duplicates.add(head)

            head = head.next

        return False

    def _155_minStack(self) -> MinStack:
        return MinStack()

    def _160_getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A = {}
        while headA:
            A[headA] = False
            headA = headA.next

        while headB:
            if headB in A:
                return headB
            headB = headB.next

        return None

    def _167_twoSum(self, numbers: List[int], target: int) -> List[int]:
        nums = {}

        for i, val in enumerate(numbers):
            if target - val in nums:
                return [nums[target - val] + 1, i + 1]

            if val not in nums:
                nums[val] = i

        raise ValueError("Solution not found")

    def _168_convertToTitle(self, n: int) -> str:
        s = ""
        while n > 0:
            n -= 1
            s += chr(int(n % 26) + 65)
            n = int(n / 26)
        return s[::-1]

    def _169_majorityElement(self, nums: List[int]) -> int:
        map = {}

        for num in nums:
            if num in map:
                map[num] = map[num] + 1
            else:
                map[num] = 1

        return max(map, key=map.get)

    def _171_titleToNumber(self, s: str) -> int:
        num = 0

        for sChar in s:
            num = num * 26 + ord(sChar) - 64

        return num

    def _172_trailingZeroes(self, n: int) -> int:
        count = 0

        while n > 0:
            n = int(n / 5)
            count += n

        return count

    def _189_rotate(self, nums: List[int], k: int) -> None:
        k = int(k % len(nums))  # // example - k=7 and nums.length=3 then k=1, remove empty loops

        self._189_rotate_reverse(nums, 0, len(nums) - 1)
        self._189_rotate_reverse(nums, 0, k - 1)
        self._189_rotate_reverse(nums, k, len(nums) - 1)

    def _189_rotate_reverse(self, nums: List[int], l: int, r: int) -> None:
        while l < r:
            tmp = nums[l]
            nums[l] = nums[r]
            nums[r] = tmp

            l += 1
            r -= 1

    def _198_rob(self, nums: List[int]) -> int:
        oneHouseBefore = 0
        twoHouseBefore = 0

        for i in range(0, len(nums)):
            tmp = oneHouseBefore
            oneHouseBefore = max(twoHouseBefore + nums[i], oneHouseBefore)
            twoHouseBefore = tmp

        return oneHouseBefore

    def _202_isHappy(self, n: int) -> bool:
        if n <= 0:
            return False

        setNums = set()

        while n not in setNums:
            setNums.add(n)

            sumOfSquare = 0
            while n > 0:
                sumOfSquare += int(n % 10) * int(n % 10)
                n /= 10
            if sumOfSquare == 1:
                return True

            n = sumOfSquare

        return False

    def _203_removeElements(self, head: ListNode, val: int) -> ListNode:
        dummyHead = ListNode(-1)
        current = dummyHead

        while head:
            if head.val == val:
                head = head.next
            else:
                current.next = head
                current = current.next

                head = head.next

        current.next = None

        return dummyHead.next

    def _205_isIsomorphic(self, s: str, t: str) -> bool:
        map = {}

        for i in range(0, len(s)):
            if s[i] not in map.keys():
                if t[i] in map.values():
                    return False
                map[s[i]] = t[i]

            if map[s[i]] != t[i]:
                return False

        return True

    def _217_containsDuplicate(self, nums: List[int]) -> bool:
        if nums is None:
            return False

        setNums = set()

        for num in nums:
            if num in setNums:
                return True
            setNums.add(num)

        return False

    def _219_containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        map = {}

        for i, num in enumerate(nums):
            if num in map.keys() and abs(i - map.get(num) <= k):
                return True

            map[num] = i

        return False

    def _226_invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None

        queue = deque()
        queue.append(root)

        while queue:
            tn = queue.popleft()

            tmp = tn.left
            tn.left = tn.right
            tn.right = tmp

            if tn.left:
                queue.append(tn.left)
            if tn.right:
                queue.append(tn.right)

        return root

    def _231_isPowerOfTwo(self, n: int) -> bool:
        if n <= 0:
            return False

        while n % 2 == 0:
            n /= 2

        return n == 1

    def _234_isPalindrome(self, head: ListNode) -> bool:
        slow = head
        fast = head

        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next

        if fast != None:  # odd
            slow = slow.next

        fast = head
        slow = self._234_isPalindrome_reverse(slow)

        while fast != None and slow != None:
            if fast.val != slow.val:
                return False

            fast = fast.next
            slow = slow.next

        return True

    def _234_isPalindrome_reverse(self, head: ListNode) -> ListNode:
        prev = None
        current = head
        next = None

        while current:
            # store next node
            next = current.next

            # this is where actual reversing happens
            current.next = prev

            # move prev and curr one step forward
            prev = current
            current = next

        head = prev

        return prev

    def _283_moveZeroes(self, nums: List[int]) -> None:
        slow = 0
        for i in range(0, len(nums)):
            if nums[i] == 0:
                continue
            nums[slow] = nums[i]
            slow += 1

        while slow < len(nums):
            nums[slow] = 0
            slow += 1

    def _344_reverseString(self, s: List[str]) -> None:
        l = 0
        r = len(s) - 1

        while l < r:
            tmp = s[l]
            s[l] = s[r]
            s[r] = tmp

            l += 1
            r -= 1

    def _509_fib(self, N: int) -> int:
        if N <= 1:
            return N

        f0 = 0
        f1 = 1
        f2 = 0

        for i in range(2, N + 1):
            f2 = f0 + f1
            f0 = f1
            f1 = f2

        return f2

    _543_max_diameter = 0

    def _543_diameterOfBinaryTree(self, root: TreeNode) -> int:
        self._543_diameterOfBinaryTree_maxDepth(root)

        return self._543_max_diameter

    def _543_diameterOfBinaryTree_maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        lMaxDepth = self._543_diameterOfBinaryTree_maxDepth(root.left)
        rMaxDepth = self._543_diameterOfBinaryTree_maxDepth(root.right)

        self._543_max_diameter = max(self._543_max_diameter, lMaxDepth + rMaxDepth)

        return 1 + max(lMaxDepth, rMaxDepth)

    def _557_reverseWords(self, s: str) -> str:
        result = ''
        ss = s.split(' ')

        for word in ss:
            result += word[::-1] + ' '

        result = result[0:len(result) - 1]

        return result

    def _561_arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()

        sum = 0
        for i in range(0, len(nums), 2):
            sum += nums[i]

        return sum

    def _589_preorder(self, root: Node) -> List[int]:
        if root == None:
            return []

        traverse = []

        stack = deque()
        stack.append(root)

        while stack:
            tn = stack.pop()

            traverse.append(tn.val)

            if tn.children:
                for child in reversed(tn.children):
                    stack.append(child)

        return traverse

    def _590_postorder(self, root: Node) -> List[int]:
        if root == None:
            return []

        traverse = []

        stack1 = deque()
        stack2 = deque()

        stack1.append(root)

        while stack1:
            tn = stack1.pop()
            stack2.append(tn)

            if tn.children:
                for child in tn.children:
                    stack1.append(child)

        while stack2:
            traverse.append(stack2.pop().val)

        return traverse

    def _617_mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if t1 == None and t2 == None:
            return None

        sum = (t1.val if t1 != None else 0) + (t2.val if t2 != None else 0)
        tn = TreeNode(sum)
        tn.left = self._617_mergeTrees((t1.left if t1 != None else None), (t2.left if t2 != None else None))
        tn.right = self._617_mergeTrees((t1.right if t1 != None else None), (t2.right if t2 != None else None))

        return tn

    def _657_judgeCircle(self, moves: str) -> bool:
        position = [0, 0]

        for move in moves:
            if move == 'L':
                position[0] -= 1
            if move == 'R':
                position[0] += 1
            if move == 'U':
                position[1] += 1
            if move == 'D':
                position[1] -= 1

        return position[0] == 0 and position[1] == 0

    def _665_checkPossibility(self, nums: List[int]) -> bool:
        modyfications = 0
        for i in range(1, len(nums)):
            if (nums[i] < nums[i - 1]):
                modyfications += 1
                if (i - 2 >= 0 and nums[i] < nums[i - 2]):
                    nums[i] = nums[i - 1]

        return modyfications <= 1

    def _690_getImportance(self, employees: List[Employee], id: int) -> int:
        map = {}

        for emp in employees:
            map[emp.id] = emp

        return self._690_getImportance_sum(map, id)

    def _690_getImportance_sum(self, map: Dict, id: int):
        sum = map[id].importance

        for subordinate in map[id].subordinates:
            sum += self._690_getImportance_sum(map, subordinate)

        return sum

    def _700_searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if val < root.val:
                root = root.left
            elif val > root.val:
                root = root.right
            else:
                return root

        return None

    def _705_myHashSet(self) -> MyHashSet:
        return MyHashSet()

    def _709_toLowerCase(self, str: str) -> str:
        if str == None:
            return None
        if len(str) == 0:
            return ""

        lStr = []

        for i in range(0, len(str)):
            value = ord(str[i])
            if value >= 65 and value <= 90:
                lStr.append(chr(value + 32))
            else:
                lStr.append(str[i])
            i += 1

        return ''.join(lStr)

    def _728_selfDividingNumbers(self, left: int, right: int) -> List[int]:
        numbers = []
        for number in range(left, right + 1):
            tmp = number
            noSelfDividing = 0
            while tmp > 0:
                digit = tmp % 10
                tmp = int(tmp / 10)

                if digit == 0 or number % digit != 0:
                    noSelfDividing += 1
                    break

            if noSelfDividing == 0:
                numbers.append(number)

        return numbers

    def _771_numJewelsInStones(self, J: str, S: str) -> int:
        map = {}

        for stone in S:
            map[stone] = 1 + map.get(stone, 0)

        count = 0

        for jewel in J:
            count += map.get(jewel, 0)

        return count

    def _796_rotateString(self, A: str, B: str) -> bool:
        if len(A) != len(B):
            return False

        return B in (A + A)

    def _804_uniqueMorseRepresentations(self, words: List[str]) -> int:
        MORSE = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.",
                 "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]

        hashSet = set()

        for word in words:
            encoded = ''
            for cChar in word:
                encoded += MORSE[ord(cChar) - 97]

            hashSet.add(encoded)

        return len(hashSet)

    def _811_subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        map = {}

        for cpdomain in cpdomains:
            scpdomain = cpdomain.split(' ')
            count = int(scpdomain[0])
            domains = scpdomain[1]

            map[domains] = map.get(domains, 0) + count
            while domains.find(".") != -1:
                domains = domains[domains.find(".") + 1: len(domains)]
                map[domains] = map.get(domains, 0) + count

        results = []
        for key, value in map.items():
            results.append(str(value) + " " + str(key))

        return results

    def _821_shortestToChar(self, S: str, C: str) -> List[int]:
        positions = []

        cIndexes = set()

        for i in range(0, len(S)):
            if S[i] == C:
                cIndexes.add(i)

        for i in range(0, len(S)):
            nearestIndex = min([abs(index - i) for index in cIndexes])
            positions.append(nearestIndex)

        return positions

    def _832_flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        rLength = len(A)
        cLength = len(A[0])
        B = [[None for i in range(cLength)] for j in range(rLength)]

        for row in range(0, rLength):
            for bCol, aCol in zip(range(0, cLength), reversed(range(0, cLength))):
                B[row][bCol] = 1 - A[row][aCol]

        return B

    def _844_backspaceCompare(self, S: str, T: str) -> bool:
        stackS = deque()
        stackT = deque()

        for cChar in S:
            if stackS and cChar == '#':
                stackS.pop()

            if cChar != '#':
                stackS.append(cChar)

        for cChar in T:
            if stackT and cChar == '#':
                stackT.pop()

            if cChar != '#':
                stackT.append(cChar)

        return ''.join(stackS) == ''.join(stackT)

    def _852_peakIndexInMountainArray(self, A: List[int]) -> int:
        l = 0
        h = len(A) - 1

        while l < h:
            m = l + int((h - l) / 2)

            if A[m] < A[m + 1]:
                l = m + 1
            else:
                h = m

        return l

    def _876_middleNode(self, head: ListNode) -> ListNode:
        slow = head
        fast = head

        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next

        return slow

    def _905_sortArrayByParity(self, A: List[int]) -> List[int]:
        B = [None for i in range(len(A))]
        k = 0
        odd = len(A) - 1
        even = 0

        while k < len(A):
            if A[k] % 2 == 0:
                B[even] = A[k]
                even += 1
            else:
                B[odd] = A[k]
                odd -= 1
            k += 1

        return B

    def _922_sortArrayByParityII(self, A: List[int]) -> List[int]:
        B = [None for i in range(len(A))]
        k = 0
        odd = 1
        even = 0

        while k < len(A):
            if A[k] % 2 == 0:
                B[even] = A[k]
                even += 2
            else:
                B[odd] = A[k]
                odd += 2
            k += 1

        return B

    def _929_numUniqueEmails(self, emails: List[str]) -> int:
        hashSet = set()

        for email in emails:
            sEmail = email.split("@")

            sEmail[0] = sEmail[0] if "+" not in sEmail[0] else sEmail[0][0:sEmail[0].index('+')]
            sEmail[0] = sEmail[0].replace('.', '')

            hashSet.add(sEmail[0] + "@" + sEmail[1])

        return len(hashSet)

    def _937_reorderLogFiles(self, logs: List[str]) -> List[str]:
        return sorted(logs, key=cmp_to_key(lambda v1, v2: self._937_reorderLogFiles_compartor(v1, v2)))

    def _937_reorderLogFiles_compartor(self, log1: str, log2: str) -> int:
        split1 = log1.split(" ", 2)
        split2 = log2.split(" ", 2)

        if split1[1][0].isalpha() and split2[1][0].isalpha():
            cmp = (split1[1] > split2[1]) - (split1[1] < split2[1])
            if cmp != 0:
                return cmp
            else:
                return (split1[0] > split2[0]) - (split1[0] < split2[0])
        elif split1[1][0].isdigit() and split2[1][0].isdigit():
            return 0
        else:
            if split1[1][0].isalpha():
                return -1
            else:
                return 1
        return 0

    def _938_rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
        if root is None:
            return 0

        queue = deque()
        queue.append(root)

        sum = 0

        while queue:
            tn = queue.popleft()

            if tn.val >= L and tn.val <= R:
                sum += tn.val

            if tn.left is not None:
                queue.append(tn.left)
            if tn.right is not None:
                queue.append(tn.right)

        return sum

    def _942_diStringMatch(self, S: str) -> List[int]:
        values = []
        increase = 0
        decrease = len(S)

        for cChar in S:
            if cChar == 'I':
                values.append(increase)
                increase += 1
            elif cChar == 'D':
                values.append(decrease)
                decrease -= 1

        values.append(increase)

        return values

    def _944_minDeletionSize(self, A: List[str]) -> int:
        count = 0

        for col in range(0, len(A[0])):
            for row in range(1, len(A)):
                if A[row - 1][col] > A[row][col]:
                    count += 1
                    break

        return count

    def _961_repeatedNTimes(self, A: List[int]) -> int:
        N = int(len(A) / 2)
        hashMap = {}

        for a in A:
            hashMap[a] = hashMap.get(a, 0) + 1

        return next(iter([key for (key, value) in hashMap.items() if value == N]), None)

    def _977_sortedSquares(self, A: List[int]) -> List[int]:
        B = [None for i in range(len(A))]
        l = 0
        r = len(A) - 1
        i = r

        while l <= r:
            if A[l] * A[l] > A[r] * A[r]:
                B[i] = A[l] * A[l]
                l += 1
            else:
                B[i] = A[r] * A[r]
                r -= 1

            i -= 1

        return B

    def _1002_commonChars(self, A: List[str]) -> List[str]:
        B = []

        for commonChar in ascii_lowercase:
            minCount = 0x7fffffff  # hex(-2**31-1) Integer.MIN_VALUE

            for a in A:
                count = 0
                for cChar in a:
                    if cChar == commonChar:
                        count += 1
                minCount = min(minCount, count)

            while minCount > 0:
                B.append(commonChar)
                minCount -= 1

        return B

    def _1021_removeOuterParentheses(self, S: str) -> str:
        res = []
        opened = 0

        for c in S:
            if c == '(' and opened > 0:
                res.append(c)
            if c == ')' and opened > 1:
                res.append(c)
            opened += 1 if c == '(' else -1

        return "".join(res)

    def _1046_lastStoneWeight(self, stones: List[int]) -> int:
        queue = KeyPriorityQueue(key=cmp_to_key(lambda o1, o2: o2 - o1))

        for stone in stones:
            queue.put(stone)

        while queue.qsize() >= 2:
            stone1 = queue.get()
            stone2 = queue.get()
            stone3 = abs(stone2 - stone1)

            if stone3 != 0:
                queue.put(stone3)

        return queue.get() if not queue.empty() else 0

    def _1047_removeDuplicates(self, S: str) -> str:
        if S is None:
            return None

        stack = deque()

        for c in S:
            if stack and stack[-1] == c:
                stack.pop()
            else:
                stack.append(c)

        return ''.join(stack)

    def _1051_heightChecker(self, heights: List[int]) -> int:
        sHeights = sorted(heights)
        count = 0

        for i in range(0, len(heights)):
            count += 1 if heights[i] != sHeights[i] else 0

        return count

    def _1078_findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        sText = text.split(" ")
        words = []

        for i in range(2, len(sText)):
            if sText[i - 2] == first and sText[i - 1] == second:
                words.append(sText[i])
            i += 1

        return words

    def _1089_duplicateZeros(self, arr: List[int]) -> None:
        i = 0
        while i < len(arr):
            if arr[i] == 0:
                j = len(arr) - 1
                while j > i:
                    arr[j] = arr[j - 1]
                    j -= 1

                if i + 1 < len(arr):
                    arr[i + 1] = 0
                i += 1
            i += 1

    def _1108_defangIPaddr(self, address: str) -> str:
        return address.replace(".", "[.]")

    def _1122_relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        arr = []
        map = {}

        for val in arr1:
            map[val] = map.get(val, 0) + 1

        for i in range(len(arr2)):
            count = map[arr2[i]]
            while count > 0:
                arr.append(arr2[i])
                count -= 1

        arr1 = list(filter(lambda x: x not in arr2, arr1))
        arr1.sort()

        for val in arr1:
            arr.append(val)

        return arr

    def _1154_dayOfYear(self, date: str) -> int:
        sDate = date.split("-")

        year = int(sDate[0])
        month = int(sDate[1])
        day = int(sDate[2])

        leapYear = True if year % 4 == 0 and year % 100 != 0 or year % 400 == 0 else False

        days = [31, 28 + leapYear, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        dayOfYear = 0

        for i in range(0, month - 1):
            dayOfYear += days[i]

        return dayOfYear + day

    def _1160_countCharacters(self, words: List[str], chars: str) -> int:
        counter = 0

        for word in words:
            tmpWord = ""
            tmpChars = chars

            for cChar in word:
                if cChar in tmpChars:
                    tmpChars = tmpChars.replace(cChar, "", 1)
                    tmpWord += cChar

            if word == tmpWord:
                counter += len(word)

        return counter

    def _1360_daysBetweenDates(self, date1: str, date2: str) -> int:
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        sDate1 = date1.split("-")
        sDate2 = date2.split("-")

        year1 = int(sDate1[0])
        month1 = int(sDate1[1])
        day1 = int(sDate1[2])

        year2 = int(sDate2[0])
        month2 = int(sDate2[1])
        day2 = int(sDate2[2])

        days1 = year1 * 365 + day1
        days2 = year2 * 365 + day2

        for i in range(month1 - 1):
            days1 += days[i]

        for i in range(month2 - 1):
            days2 += days[i]

        if month1 <= 2:
            year1 -= 1
        days1 += int(year1 / 4) - int(year1 / 100) + int(year1 / 400)

        if month2 <= 2:
            year2 -= 1
        days2 += int(year2 / 4) - int(year2 / 100) + int(year2 / 400)

        return abs(days1 - days2)

    def _1365_smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        sNums = nums.copy()
        sNums.sort()

        map = {}

        for i, val in enumerate(sNums):
            map.setdefault(val, i)

        values = []

        for val in nums:
            values.append(map.get(val))

        return values

    def _1374_generateTheString(self, n: int) -> str:
        return "a" * (n - 1) + "b" if n % 2 == 0 else "a" * n

    def _1380_luckyNumbers(self, matrix: List[List[int]]) -> List[int]:
        minRow = set([min(row) for row in matrix])
        maxCol = set()

        for col in range(len(matrix[0])):
            maxCol.add([max(row) for row in zip(*matrix)][col])

        return list(minRow & maxCol)

    def _1389_createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        values = []

        for i in range(len(nums)):
            values.insert(index[i], nums[i])

        return values

    def _1394_findLucky(self, arr: List[int]) -> int:
        map = {}

        for val in arr:
            map[val] = map.get(val, 0) + 1

        luckyNumbers = [k for k, v in map.items() if k == v]

        return max(luckyNumbers) if luckyNumbers else -1

    def _1399_countLargestGroup(self, n: int) -> int:
        countGroups = {}

        for i in range(1, n + 1):
            sum = 0
            while i > 0:
                sum += i % 10
                i = int(i / 10)
            countGroups[sum] = countGroups.get(sum, 0) + 1

        maxValue = max(countGroups.values())

        return len(list(filter(lambda x: x == maxValue, countGroups.values())))

    def _1403_minSubsequence(self, nums: List[int]) -> List[int]:
        if nums == None:
            return None

        sumNums = sum(nums)

        nums.sort(reverse=True)

        values = []
        subSumNums = 0
        for val in nums:
            subSumNums += val
            values.append(val)

            if subSumNums > sumNums - subSumNums:
                return values

        return values


def main():
    pass


if __name__ == "__main__":
    main()
