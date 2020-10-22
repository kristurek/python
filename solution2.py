import collections
from collections import deque
from typing import List, Dict

from common import ListNode, TreeNode, GraphNode
from common import Node2


class Node3:
    def __init__(self, key: int, val: int):
        self.key = key
        self.val = val
        self.next = None
        self.previous = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {}
        self.head = Node3(0, 0)
        self.tail = Node3(0, 0)

        self.head.next = self.tail
        self.tail.previous = self.head

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1

        node = self.map[key]

        self.removeNode(node)
        self.addNode(node)

        return node.val

    def put(self, key: int, value: int) -> None:
        if len(self.map) < self.capacity:
            if key not in self.map:
                node = Node3(key, value)

                self.map[key] = node
                self.addNode(node)
            else:
                node = self.map[key]

                node.val = value
                self.removeNode(node)
                self.addNode(node)
        else:
            if key not in self.map:
                node = Node3(key, value)

                self.map.pop(self.tail.previous.key)
                self.removeNode(self.tail.previous)

                self.map[key] = node
                self.addNode(node)
            else:
                node = self.map[key]

                node.val = value

                self.removeNode(node)
                self.addNode(node)

    def addNode(self, node: Node3) -> None:
        after = self.head.next

        self.head.next = node
        node.next = after
        node.previous = self.head
        after.previous = node

    def removeNode(self, node: Node3) -> None:
        previous = node.previous
        next = node.next

        previous.next = next
        next.previous = previous


class Solution:

    def _0_simple(self) -> bool:
        return True

    def _1_twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}

        for i, v in enumerate(nums):
            val = target - v
            if val in map:
                return [map[val], i]
            else:
                map[v] = i

        raise ValueError("No solution found")

    def _2_addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(-1)
        l3 = head

        carry = 0
        while l1 != None or l2 != None:
            v1 = l1.val if l1 != None else 0
            v2 = l2.val if l2 != None else 0

            sum = v1 + v2 + carry

            l3.next = ListNode(sum % 10)
            l3 = l3.next

            carry = sum // 10

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

        for cChar in s:
            if cChar not in queue:
                queue.append(cChar)
                maxLength = max(maxLength, len(queue))
            else:
                while cChar in queue:
                    queue.popleft()
                queue.append(cChar)

        return maxLength

    def _4_findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums = []

        j = 0
        k = 0
        for i in range(0, len(nums1) + len(nums2)):
            if j < len(nums1) and k < len(nums2):
                if nums1[j] < nums2[k]:
                    nums.append(nums1[j])
                    j += 1
                else:
                    nums.append(nums2[k])
                    k += 1
            elif j < len(nums1):
                nums.append(nums1[j])
                j += 1
            else:
                nums.append(nums2[k])
                k += 1

        if len(nums) % 2 == 0:
            return (nums[len(nums) // 2] + nums[len(nums) // 2 - 1]) / 2
        else:
            return nums[len(nums) // 2]

    def _5_longestPalindrome(self, s: str) -> str:
        if s is None or len(s) <= 1:
            return s

        maxP = ""

        for i in range(0, len(s)):
            tmp = self._5_longestPalindrome_extends(s, i, i)
            maxP = tmp if len(maxP) < len(tmp) else maxP

            tmp = self._5_longestPalindrome_extends(s, i, i + 1)
            maxP = tmp if len(maxP) < len(tmp) else maxP

        return maxP

    def _5_longestPalindrome_extends(self, s: str, begin, end) -> str:
        while begin >= 0 and end < len(s) and s[begin] == s[end]:
            begin -= 1
            end += 1

        return s[begin + 1: end]

    def _7_reverse(self, x: int) -> int:
        min_integer = -0x80000000  # hex(-2**31-1) Integer.MIN_VALUE
        max_integer = 0x7fffffff  # hex(2**31-1) Integer.MAX_VALUE
        number = 0
        minus = -1 if x < 0 else 1
        x = abs(x)

        while x > 0:
            number = number * 10 + x % 10
            x = x // 10

        if number > max_integer or number < min_integer:
            return 0

        return number * minus

    def _9_isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False

        return self._7_reverse(x) == x

    def _11_maxArea(self, height: List[int]) -> int:
        min_integer = -0x80000000  # hex(-2**31-1) Integer.MIN_VALUE

        maxArea = min_integer
        l = 0
        h = len(height) - 1

        while l < h:
            localArea = (h - l) * min(height[l], height[h])

            maxArea = max(localArea, maxArea)

            if height[l] < height[h]:
                l += 1
            else:
                h -= 1

        return maxArea

    def _12_intToRoman(self, num: int) -> str:
        map = collections.OrderedDict()
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

        number = ""

        for k, v in map.items():
            while num // k > 0:
                number += v
                num = num - k

        return number

    def _13_romanToInt(self, s: str) -> int:
        map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000, }

        number = 0

        for i in range(0, len(s) - 1):
            current = map[s[i]]
            next = map[s[i + 1]]

            if current < next:
                number -= current
            else:
                number += current

        return number + map[s[len(s) - 1]]

    def _14_longestCommonPrefix(self, strs: List[str]) -> str:
        longestPrefix = strs[0]

        for i in reversed(range(0, len(longestPrefix))):
            for str in strs:
                if i >= len(str) or str[i] != longestPrefix[i]:
                    longestPrefix = longestPrefix[0:i]
                    break

        return longestPrefix

    def _15_threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        groups = set()

        for i in range(0, len(nums)):
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
        max_integer = 0x7fffffff  # hex(2**31-1) Integer.MAX_VALUE

        nums = sorted(nums)
        closestSum = max_integer

        for i in range(0, len(nums)):
            l = i + 1
            h = len(nums) - 1

            while l < h:
                sum = nums[i] + nums[l] + nums[h]

                if abs(target - sum) < abs(target - closestSum):
                    closestSum = sum

                if sum < target:
                    l += 1
                elif sum > target:
                    h -= 1
                else:
                    return target

        return closestSum

    def _17_letterCombinations(self, digits: str) -> List[str]:
        if digits is None or len(digits) == 0:
            return []

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

        output = []

        self._17_letterCombinations_backtracking(output, digits, [], 0, map)

        return output

    def _17_letterCombinations_backtracking(self, output: List[str], digits: str, tmp: List[str], index: int,
                                            map: Dict) -> None:
        if len(tmp) == len(digits):
            output.append("".join(tmp))
        else:
            for cChar in map[digits[index]]:
                tmp.append(cChar)
                self._17_letterCombinations_backtracking(output, digits, tmp, index + 1, map)
                tmp.pop()

    def _19_removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        nodes = []

        current = head

        while current is not None:
            nodes.append(current)
            current = current.next

        if len(nodes) - n == 0:
            return head.next
        else:
            node = nodes[len(nodes) - n - 1]
            node.next = node.next.next
            return head

    def _20_isValid(self, s: str) -> bool:
        stack = []

        for bracket in s:
            if bracket == '(' or bracket == '{' or bracket == '[':
                stack.append(bracket)
            elif bracket == ')' and stack and stack[- 1] == '(':
                stack.pop()
            elif bracket == '}' and stack and stack[- 1] == '{':
                stack.pop()
            elif bracket == ']' and stack and stack[- 1] == '[':
                stack.pop()
            else:
                return False

        return len(stack) == 0

    def _21_mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(-1)
        l3 = head

        while l1 != None and l2 != None:
            if l1.val < l2.val:
                l3.next = l1

                l1 = l1.next
                l3 = l3.next
            else:
                l3.next = l2

                l2 = l2.next
                l3 = l3.next

        if l1 != None:
            l3.next = l1

        if l2 != None:
            l3.next = l2

        return head.next

    def _22_generateParenthesis(self, n: int) -> List[str]:
        output = []

        self._22_generateParenthesis_backtracking(output, "", n, 0, 0)

        return output

    def _22_generateParenthesis_backtracking(self, output: List[str], tmp: str, n: int, open: int, close: int) -> None:
        if len(tmp) == 2 * n:
            output.append(tmp)
        else:
            if open >= close:
                if open < n:
                    self._22_generateParenthesis_backtracking(output, tmp + "(", n, open + 1, close)
                if close < n:
                    self._22_generateParenthesis_backtracking(output, tmp + ")", n, open, close + 1)

    def _23_mergeKLists(self, lists: List[ListNode]) -> ListNode:
        return self._23_mergeKLists_divide(lists, 0, len(lists) - 1)

    def _23_mergeKLists_divide(self, lists: List[ListNode], l: int, r: int) -> ListNode:
        if l == r:
            return lists[l]
        elif l < r:
            m = l + (r - l) // 2

            l1 = self._23_mergeKLists_divide(lists, l, m)
            l2 = self._23_mergeKLists_divide(lists, m + 1, r)

            return self._23_mergeKLists_merge(l1, l2)
        else:
            return None

    def _23_mergeKLists_merge(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(-1)
        l3 = head

        while l1 != None and l2 != None:
            if l1.val < l2.val:
                l3.next = l1

                l3 = l3.next
                l1 = l1.next
            else:
                l3.next = l2

                l3 = l3.next
                l2 = l2.next

        if l1 != None:
            l3.next = l1
        if l2 != None:
            l3.next = l2

        return head.next

    def _24_swapPairs(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        dummy = ListNode(-1)
        current = dummy

        slow = head
        fast = head.next

        while fast != None:
            tmp = fast.next

            fast.next = None
            slow.next = None

            current.next = fast
            current = current.next
            current.next = slow
            current = current.next

            slow = tmp
            fast = tmp.next if tmp != None else None

        if slow != None:
            current.next = slow

        return dummy.next

    def _26_removeDuplicates(self, nums: List[int]) -> int:
        slow = 0
        fast = 0

        while fast < len(nums):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]

            fast += 1

        return slow + 1

    def _27_removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        fast = 0

        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        return slow

    def _28_strStr(self, haystack: str, needle: str) -> int:
        if haystack == None or needle == None:
            return -1

        if len(needle) == 0:
            return 0

        i = 0
        j = 0

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
        l = 0
        h = len(nums) - 1

        while l < h:
            m = l + (h - l) // 2

            if nums[m] > nums[h]:
                l = m + 1
            else:
                h = m

        pivot = l
        l = 0
        h = len(nums) - 1

        if nums[pivot] <= target <= nums[h]:
            l = pivot
        else:
            h = pivot - 1

        while l <= h:
            m = l + (h - l) // 2

            if target < nums[m]:
                h = m - 1
            elif target > nums[m]:
                l = m + 1
            else:
                return m

        return -1

    def _34_searchRange(self, nums: List[int], target: int) -> List[int]:
        l = self._34_searchRange_lr(nums, target, True)
        r = self._34_searchRange_lr(nums, target, False)

        return [l, r]

    def _34_searchRange_lr(self, nums: List[int], target: int, left: bool) -> int:
        l = 0
        h = len(nums) - 1

        while l <= h:
            m = l + (h - l) // 2

            if target < nums[m]:
                h = m - 1
            elif target > nums[m]:
                l = m + 1
            else:
                if left:
                    if m > 0 and nums[m] == nums[m - 1]:
                        h = m - 1
                    else:
                        return m
                else:
                    if m < len(nums) - 1 and nums[m] == nums[m + 1]:
                        l = m + 1
                    else:
                        return m
        return -1

    def _35_searchInsert(self, nums: List[int], target: int) -> int:
        l = 0
        h = len(nums) - 1

        while l <= h:
            m = l + (h - l) // 2

            if target < nums[m]:
                h = m - 1
            elif target > nums[m]:
                l = m + 1
            else:
                return m

        return l

    def _36_isValidSudoku(self, board: List[List[str]]) -> bool:
        mSet = set()
        for row in range(0, len(board)):
            for col in range(0, len(board[row])):
                if board[row][col].isdigit():
                    r = 'R' + str(row) + board[row][col]
                    c = 'C' + str(col) + board[row][col]
                    b = 'B' + str(row // 3) + str(col // 3) + board[row][col]

                    if r in mSet or c in mSet or b in mSet:
                        return False

                    mSet.add(r)
                    mSet.add(c)
                    mSet.add(b)
        return True

    def _39_combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        output = []

        self._39_combinationSum_backtracking(output, [], candidates, target, 0)

        return output

    def _39_combinationSum_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int], remain: int,
                                        begin: int):
        if remain == 0:
            output.append(list(tmp))
        elif remain < 0:
            return
        else:
            for i in range(begin, len(nums)):
                tmp.append(nums[i])
                self._39_combinationSum_backtracking(output, tmp, nums, remain - nums[i], i)
                tmp.pop()

    def _40_combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        output = []
        candidates = sorted(candidates)
        self._40_combinationSum2_backtracking(output, [], candidates, target, 0)

        return output

    def _40_combinationSum2_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int], remain, begin):
        if remain == 0:
            output.append(list(tmp))
        elif remain < 0:
            return
        else:
            for i in range(begin, len(nums)):
                if i > begin and nums[i] == nums[i - 1]:
                    continue

                tmp.append(nums[i])
                self._40_combinationSum2_backtracking(output, tmp, nums, remain - nums[i], i + 1)
                tmp.pop()

    def _41_firstMissingPositive(self, nums: List[int]) -> int:
        nums = sorted(nums)

        res = 1

        for num in nums:
            if num == res:
                res += 1

        return res

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
        used = [False for i in range(0, len(nums))]

        nums = sorted(nums)

        self._47_permuteUnique_backtracking(output, [], nums, used)

        return output

    def _47_permuteUnique_backtracking(self, output: List[List[int]], tmp: List[int], nums: List[int],
                                       used: List[bool]) -> None:
        if len(tmp) == len(nums):
            output.append(list(tmp))
        else:
            for i in range(0, len(nums)):
                if used[i] == True or (i > 0 and nums[i] == nums[i - 1] and used[i - 1] == True):
                    continue

                tmp.append(nums[i])
                used[i] = True
                self._47_permuteUnique_backtracking(output, tmp, nums, used)
                used[i] = False
                tmp.pop()

    def _49_groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        map = {}

        for lStr in strs:
            key = "".join(sorted(lStr))

            values = map.get(key, [])
            values.append(lStr)

            map[key] = values

        return [item[1] for item in map.items()]

    def _50_myPow(self, x: float, n: int) -> float:
        if n < 0:
            return 1 / self._50_myPow_helper(x, -n)
        else:
            return self._50_myPow_helper(x, n)

    def _50_myPow_helper(self, x: float, n: int) -> float:
        if n == 0:
            return 1

        tmp = self._50_myPow_helper(x, int(n / 2))

        if n % 2 == 0:
            return tmp * tmp
        else:
            return (tmp * tmp) * x

    def _53_maxSubArray(self, nums: List[int]) -> int:
        min_integer = -0x80000000  # hex(-2**31-1) Integer.MIN_VALUE
        max = min_integer
        sum = 0

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

        colBegin = 0
        colEnd = len(matrix[0]) - 1
        rowBegin = 0
        rowEnd = len(matrix) - 1

        while colBegin <= colEnd and rowBegin <= rowEnd:
            for col in range(colBegin, colEnd + 1):
                values.append(matrix[rowBegin][col])
            rowBegin += 1

            for row in range(rowBegin, rowEnd + 1):
                values.append(matrix[row][colEnd])
            colEnd -= 1

            if rowBegin <= rowEnd:
                for col in reversed(range(colBegin, colEnd + 1)):
                    values.append(matrix[rowEnd][col])
                rowEnd -= 1

            if colBegin <= colEnd:
                for row in reversed \
                            (range(rowBegin, rowEnd + 1)):
                    values.append(matrix[row][colBegin])
                colBegin += 1

        return values

    def _58_lengthOfLastWord(self, s: str) -> int:
        s = s.strip()
        ss = s.split(" ")

        return len(ss[len(ss) - 1])

    def _61_rotateRight(self, head: ListNode, k: int) -> ListNode:
        curr = head

        length = 1
        while curr.next != None:
            length += 1
            curr = curr.next

        k = k % length

        curr.next = head

        for i in range(0, length - k):
            curr = curr.next

        head = curr.next
        curr.next = None

        return head

    def _66_plusOne(self, digits: List[int]) -> List[int]:
        for i in reversed(range(0, len(digits))):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1
                return digits

        digits = [0] * (len(digits) + 1)
        digits[0] = 1

        return digits

    def _67_addBinary(self, a: str, b: str) -> str:
        c = []
        i = len(a) - 1
        j = len(b) - 1

        carry = 0
        while i >= 0 or j >= 0:
            sum = carry
            sum += int(a[i]) if i >= 0 else 0
            sum += int(b[j]) if j >= 0 else 0

            c.append(str(sum % 2))
            carry = sum // 2

            i -= 1
            j -= 1

        if carry != 0:
            c.append('1')

        return "".join(reversed(c))

    def _69_mySqrt(self, x: int) -> int:
        l = 0
        h = 2 * x
        answer = -1

        while l <= h:
            m = l + (h - l) // 2

            if m * m < x:
                l = m + 1
                answer = m
            elif m * m > x:
                h = m - 1
            else:
                return m

        return answer

    def _70_climbStairs(self, n: int) -> int:
        if n <= 2:
            return n

        firstStep = 1
        secondStep = 2

        n = n - 2

        while n > 0:
            n -= 1

            tmp = firstStep
            firstStep = secondStep
            secondStep = tmp + secondStep

        return secondStep

    def _71_simplifyPath(self, path: str) -> str:
        stack = []

        for cmd in path.split("/"):
            if cmd == "..":
                if stack:
                    stack.pop()
            elif cmd == "" or cmd == "." or cmd == "/":
                continue
            else:
                stack.append(cmd)

        return "/" + "/".join(stack) if stack else "/"

    def _73_setZeroes(self, matrix: List[List[int]]) -> None:
        rows = set()
        cols = set()

        for row in range(0, len(matrix)):
            for col in range(0, len(matrix[row])):
                if matrix[row][col] == 0:
                    rows.add(row)
                    cols.add(col)

        for row in range(0, len(matrix)):
            for col in range(0, len(matrix[row])):
                if row in rows or col in cols:
                    matrix[row][col] = 0

    def _74_searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if matrix == None or len(matrix) == 0:
            return False

        l = 0
        h = len(matrix) * len(matrix[0]) - 1

        while l <= h:
            m = l + (h - l) // 2

            row = m // len(matrix[0])
            col = m % len(matrix[0])

            if target < matrix[row][col]:
                h = m - 1
            elif target > matrix[row][col]:
                l = m + 1
            else:
                return True

        return False

    def _75_sortColors(self, nums: List[int]) -> None:
        self._75_sortColors_quicksort(nums, 0, len(nums) - 1)

    def _75_sortColors_quicksort(self, nums, left, right):
        if left < right:
            splitPoint = self._75_sortColors_partition(nums, left, right)

            self._75_sortColors_quicksort(nums, left, splitPoint - 1)
            self._75_sortColors_quicksort(nums, splitPoint + 1, right)

    def _75_sortColors_partition(self, nums, left, right) -> int:
        pivot = nums[right]
        lowerIndex = left - 1

        for currentIndex in range(left, right):
            if nums[currentIndex] < pivot:  # < increase order , > decrease order
                lowerIndex += 1

                tmp = nums[currentIndex]
                nums[currentIndex] = nums[lowerIndex]
                nums[lowerIndex] = tmp

        lowerIndex += 1
        tmp = nums[lowerIndex]
        nums[lowerIndex] = nums[right]
        nums[right] = tmp

        return lowerIndex

    def _77_combine(self, n: int, k: int) -> List[List[int]]:
        nums = [i for i in range(1, n + 1)]
        output = []

        self._77_combine_backtracking(output, [], nums, 0, k)

        return output

    def _77_combine_backtracking(self, output, tmp, nums, begin, k):
        if len(tmp) == k:
            output.append(list(tmp))

        for i in range(begin, len(nums)):
            tmp.append(nums[i])
            self._77_combine_backtracking(output, tmp, nums, i + 1, k)
            tmp.pop()

    def _78_subsets(self, nums: List[int]) -> List[List[int]]:
        output = []

        self._78_subsets_backtracking(output, [], nums, 0)

        return output

    def _78_subsets_backtracking(self, output, tmp, nums, begin):
        output.append(list(tmp))

        for i in range(begin, len(nums)):
            tmp.append(nums[i])
            self._78_subsets_backtracking(output, tmp, nums, i + 1)
            tmp.pop()

    def _79_exist(self, board: List[List[str]], word: str) -> bool:
        for row in range(0, len(board)):
            for col in range(0, len(board[row])):
                if self._79_exist_search(board, row, col, list(word), 0):
                    return True

        return False

    def _79_exist_search(self, board, row, col, wChars, i):
        if i == len(wChars):
            return True

        if row < 0 or row >= len(board) or col < 0 or col >= len(board[row]):
            return False

        if board[row][col] == wChars[i]:
            tmp = board[row][col]
            board[row][col] = "*"

            isFound = self._79_exist_search(board, row + 1, col, wChars, i + 1) or \
                      self._79_exist_search(board, row - 1, col, wChars, i + 1) or \
                      self._79_exist_search(board, row, col + 1, wChars, i + 1) or \
                      self._79_exist_search(board, row, col - 1, wChars, i + 1)

            board[row][col] = tmp

            return isFound
        return False

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
            m = l + (h - l) // 2

            if target == nums[m]:
                return True
            elif nums[m] == nums[h]:
                h -= 1
            elif nums[m] == nums[l]:
                l += 1
            elif nums[m] > nums[h]:
                if target >= nums[l] and target < nums[m]:
                    h = m - 1
                else:
                    l = m + 1
            elif nums[m] < nums[h]:
                if target > nums[m] and target <= nums[h]:
                    l = m + 1
                else:
                    h = m - 1

        return False

    def _82_deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head

        dummy = ListNode(-1)

        dummy.next = head

        slow = dummy
        fast = dummy.next

        while fast != None:
            while fast.next != None and fast.val == fast.next.val:
                fast = fast.next

            if slow.next != fast:
                slow.next = fast.next
                fast = fast.next
            else:
                slow = slow.next
                fast = fast.next

        return dummy.next

    def _83_deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None:
            return head

        slow = head
        fast = head.next

        while fast != None:
            if slow.val != fast.val:
                slow = slow.next
                fast = fast.next
            else:
                slow.next = fast.next
                fast = fast.next

        return head

    def _86_partition(self, head: ListNode, x: int) -> ListNode:
        hNodeBeforeX = ListNode(-1)
        cNodeBeforeX = hNodeBeforeX
        hNodeAfterX = ListNode(-1)
        cNodeAfterX = hNodeAfterX

        while head != None:
            if head.val < x:
                cNodeBeforeX.next = head
                cNodeBeforeX = cNodeBeforeX.next
                head = head.next
            else:
                cNodeAfterX.next = head
                cNodeAfterX = cNodeAfterX.next
                head = head.next

        cNodeBeforeX.next = hNodeAfterX.next
        cNodeAfterX.next = None

        return hNodeBeforeX.next

    def _88_merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        j = m - 1
        k = n - 1
        for i in reversed(range(0, m + n)):
            if j >= 0 and k >= 0:
                if nums1[j] > nums2[k]:
                    nums1[i] = nums1[j]
                    j -= 1
                else:
                    nums1[i] = nums2[k]
                    k -= 1
            elif j >= 0:
                nums1[i] = nums1[j]
                j -= 1
            elif k >= 0:
                nums1[i] = nums2[k]
                k -= 1

    def _90_subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        output = []

        nums = sorted(nums)

        self._90_subsetsWithDup_backtracking(output, [], nums, 0)

        return output

    def _90_subsetsWithDup_backtracking(self, output, tmp, nums, begin):
        output.append(list(tmp))

        for i in range(begin, len(nums)):
            if i > begin and nums[i - 1] == nums[i]:
                continue

            tmp.append(nums[i])
            self._90_subsetsWithDup_backtracking(output, tmp, nums, i + 1)
            tmp.pop()

    def _92_reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head

        current = dummy

        for i in range(1, m):
            current = current.next

        curr = current.next
        prev = None
        next = None
        last = current.next

        for i in range(m, n + 1):
            next = curr.next

            curr.next = prev

            prev = curr
            curr = next

        current.next = prev
        last.next = curr

        return dummy.next

    def _94_inorderTraversal(self, root: TreeNode) -> List[int]:
        if root == None:
            return []

        values = []
        stack = []
        current = root

        while stack or current != None:
            if current != None:
                stack.append(current)

                current = current.left
            else:
                current = stack.pop()

                values.append(current.val)

                current = current.right

        return values

    def _98_isValidBST(self, root: TreeNode) -> bool:
        if root == None:
            return True

        stack = []
        current = root
        previous = None

        while stack or current != None:
            if current != None:
                stack.append(current)
                current = current.left
            else:
                current = stack.pop()

                if previous != None and previous.val >= current.val:
                    return False

                previous = current
                current = current.right

        return True

    def _100_isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p == None and q == None:
            return True

        if p == None or q == None:
            return False

        queue1 = deque()
        queue2 = deque()

        queue1.append(p)
        queue2.append(q)

        while queue1 and queue2:
            tn1 = queue1.popleft()
            tn2 = queue2.popleft()

            if tn1.left != None and tn2.left == None or tn1.left == None and tn2.left != None:
                return False
            if tn1.right != None and tn2.right == None or tn1.right == None and tn2.right != None:
                return False
            if tn1.val != tn2.val:
                return False

            if tn1.left != None:
                queue1.append(tn1.left)
            if tn2.right != None:
                queue2.append(tn2.left)

        return len(queue1) == 0 and len(queue2) == 0

    def _101_isSymmetric(self, root: TreeNode) -> bool:
        queue = deque()
        queue.append(root)
        queue.append(root)

        while queue:
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

    def _102_levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        levels = []
        queue = deque()
        queue.append(root)

        while queue:
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

            levels.append(level)

        return levels

    def _103_zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        levels = []
        queue = deque()
        queue.append(root)
        curLevel = 0

        while queue:
            size = len(queue)
            curLevel += 1
            level = deque()
            while size > 0:
                size -= 1

                tn = queue.popleft()

                if curLevel % 2 == 0:
                    level.appendleft(tn.val)
                else:
                    level.append(tn.val)

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

            levels.append(list(level))

        return levels

    def _104_maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0

        queue = deque()
        queue.append(root)
        curLevel = 0

        while queue:
            size = len(queue)
            curLevel += 1
            while size > 0:
                size -= 1

                tn = queue.popleft()

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

        return curLevel

    def _107_levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        levels = deque()
        queue = deque()
        queue.append(root)

        while queue:
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
        return self._108_sortedArrayToBST_build(nums, 0, len(nums) - 1)

    def _108_sortedArrayToBST_build(self, nums, l, h) -> TreeNode:
        if l <= h:
            m = l + (h - l) // 2

            tn = TreeNode(nums[m])

            tn.left = self._108_sortedArrayToBST_build(nums, l, m - 1)
            tn.right = self._108_sortedArrayToBST_build(nums, m + 1, h)

            return tn

        return None

    def _109_sortedListToBST(self, head: ListNode) -> TreeNode:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next

        return self._109_sortedListToBST_build(nums, 0, len(nums) - 1)

    def _109_sortedListToBST_build(self, nums, l, h) -> TreeNode:
        if l <= h:
            m = l + (h - l) // 2

            tn = TreeNode(nums[m])

            tn.left = self._109_sortedListToBST_build(nums, l, m - 1)
            tn.right = self._109_sortedListToBST_build(nums, m + 1, h)

            return tn

        return None

    def _110_isBalanced(self, root: TreeNode) -> bool:
        if root == None:
            return True

        heightLeft = self._110_isBalanced_height(root.left)
        heightRight = self._110_isBalanced_height(root.right)

        return abs(heightRight - heightLeft) <= 1 and self._110_isBalanced(root.left) and self._110_isBalanced(
            root.right)

    def _110_isBalanced_height(self, root):
        levels = 0

        if root == None:
            return levels

        queue = deque()
        queue.append(root)

        while queue:
            levels += 1
            size = len(queue)

            while size > 0:
                size -= 1

                tn = queue.popleft()

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

        return levels

    def _111_minDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0

        levels = 0
        queue = deque()
        queue.append(root)

        while queue:
            levels += 1
            size = len(queue)

            while size > 0:
                size -= 1

                tn = queue.popleft()

                if tn.left == None and tn.right == None:
                    return levels

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

        return levels

    def _112_hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if root == None:
            return False

        stack = deque()
        paths = deque()

        stack.append(root)
        paths.append(root.val)

        while stack:
            tn = stack.pop()
            path = paths.pop()

            if tn.left == None and tn.right == None and path == sum:
                return True

            if tn.right != None:
                stack.append(tn.right)
                paths.append(path + tn.right.val)
            if tn.left != None:
                stack.append(tn.left)
                paths.append(path + tn.left.val)

        return False

    def _113_pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        if root == None:
            return []

        validPaths = []
        stack = []
        paths = []

        stack.append(root)
        paths.append([root.val])

        while stack:
            tn = stack.pop()
            path = paths.pop()

            if tn.left == None and tn.right == None and sum(path) == target:
                validPaths.append(list(path))

            if tn.right != None:
                stack.append(tn.right)
                cPath = list(path)
                cPath.append(tn.right.val)
                paths.append(cPath)
            if tn.left != None:
                stack.append(tn.left)
                cPath = list(path)
                cPath.append(tn.left.val)
                paths.append(cPath)

        return validPaths

    def _114_flatten(self, root: TreeNode) -> None:
        if root == None:
            return

        stack = []
        previous = None

        stack.append(root)

        while stack:
            tn = stack.pop()

            left = tn.left
            right = tn.right

            tn.left = None
            tn.right = None

            if previous == None:
                previous = tn
            else:
                previous.right = tn
                previous = previous.right

            if right != None:
                stack.append(right)
            if left != None:
                stack.append(left)
        pass

    def _116_connect(self, root: 'Node') -> 'Node':
        if root == None:
            return root

        queue = deque()
        queue.append(root)

        while queue:
            size = len(queue)

            while size > 0:
                size -= 1

                tn = queue.popleft()

                tn.next = None if size == 0 else queue[0]

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

        return root

    def _117_connect(self, root: 'Node') -> 'Node':
        if root == None:
            return root

        queue = deque()
        queue.append(root)

        while queue:
            size = len(queue)

            while size > 0:
                size -= 1

                tn = queue.popleft()

                tn.next = None if size == 0 else queue[0]

                if tn.left != None:
                    queue.append(tn.left)
                if tn.right != None:
                    queue.append(tn.right)

        return root

    def _118_generate(self, numRows: int) -> List[List[int]]:
        rows = []
        if numRows == 0:
            return rows
        rows.append([1])
        if numRows == 1:
            return rows
        rows.append([1, 1])
        if numRows == 2:
            return rows

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
        rows = []

        rows.append([1])
        if rowIndex == 0:
            return rows[rowIndex]

        rows.append([1, 1])
        if rowIndex == 1:
            return rows[rowIndex]

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
        if len(prices) == 0:
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
        if len(prices) == 0:
            return 0
        max = 0

        for i in range(1, len(prices)):
            if prices[i] - prices[i - 1] > 0:
                max += prices[i] - prices[i - 1]

        return max

    def _125_isPalindrome(self, s: str) -> bool:
        s = s.lower()

        l = 0
        r = len(s) - 1

        while l < r:
            if s[l].isalpha() == False and s[l].isnumeric() == False:
                l += 1
            elif s[r].isalpha() == False and s[r].isnumeric() == False:
                r -= 1
            elif s[l] != s[r]:
                return False
            else:
                l += 1
                r -= 1

        return True

    def _129_sumNumbers(self, root: TreeNode) -> int:
        if root == None:
            return 0

        sumOfNumbers = 0

        stack = []
        sums = []

        stack.append(root)
        sums.append(root.val)

        while stack:
            tn = stack.pop()
            sum = sums.pop()

            if tn.left == None and tn.right == None:
                sumOfNumbers += sum

            if tn.right != None:
                stack.append(tn.right)
                sums.append(sum * 10 + tn.right.val)

            if tn.left != None:
                stack.append(tn.left)
                sums.append(sum * 10 + tn.left.val)

        return sumOfNumbers

    def _130_solve(self, board: List[List[str]]) -> None:
        for row in range(1, len(board) - 1):
            for col in range(1, len(board[row]) - 1):
                if board[row][col] == 'O':
                    self.check(board, row, col)

    def check(self, board, row, col):
        if row - 1 == 0 and board[row - 1][col] == 'O':
            return
        if row + 1 == len(board) - 1 and board[row + 1][col] == 'O':
            return
        if col - 1 == 0 and board[row][col - 1] == 'O':
            return
        if col + 1 == len(board[row]) - 1 and board[row][col + 1] == 'O':
            return

        if row - 1 == 0 or row + 1 == len(board) - 1 or col - 1 == 0 or col + 1 == len(board[row]) - 1:
            board[row][col] = 'X'

    def _131_partition(self, s: str) -> List[List[str]]:
        output = []

        self._131_partition_backtracking(output, [], s, 0)

        return output

    def _131_partition_backtracking(self, output, tmp, s, begin):
        if begin == len(s):
            output.append(list(tmp))

        for i in range(begin, len(s)):
            ss = s[begin: i + 1]

            if self.isPalindrome(ss):
                tmp.append(ss)
                self._131_partition_backtracking(output, tmp, s, i + 1)
                tmp.pop()

    def isPalindrome(self, ss):
        l = 0
        h = len(ss) - 1

        while l < h:
            if ss[l] != ss[h]:
                return False
            l += 1
            h -= 1

        return True

    def _133_cloneGraph(self, node: GraphNode) -> GraphNode:
        if node == None:
            return None

        map = {}
        queue = deque()

        queue.append(node)
        map[node.val] = GraphNode(node.val, [])

        while queue:
            n = queue.popleft()
            for child in n.neighbors:
                if child.val not in map:
                    map[child.val] = GraphNode(child.val, [])
                    queue.append(child)

                map[n.val].neighbors.append(map[child.val])

        return map[node.val]

    def _136_singleNumber(self, nums: List[int]) -> int:
        lSet = set()

        for num in nums:
            if num not in lSet:
                lSet.add(num)
            else:
                lSet.remove(num)

        return lSet.pop()

    def _137_singleNumber(self, nums: List[int]) -> int:
        nums = sorted(nums)
        i = 2

        while i < len(nums):
            if nums[i - 2] != nums[i]:
                return nums[i - 2]
            i = i + 3

        if len(nums) == 1:
            return nums[0]
        else:
            return nums[-1]

    def _138_copyRandomList(self, head: Node2) -> Node2:
        if head == None:
            return head

        map = {}
        current = head

        while current != None:
            map[current] = Node2(current.val)
            current = current.next

        stack = []
        stack.append(head)

        while stack:
            node = stack.pop()

            if node.next != None:
                stack.append(node.next)

            map[node].next = map.get(node.next, None)
            map[node].random = map.get(node.random, None)

        return map[head]

    def _139_wordBreak(self, s: str, wordDict: List[str]) -> bool:
        output = []

        self._139_wordBreak_backtracking(output, "", s, wordDict)

        return len(output) > 0

    def _139_wordBreak_backtracking(self, output: List[str], tmp: str, target: str, wordDict: List[str]) -> None:
        if len(tmp) > len(target):
            return
        elif tmp == target:
            output.append(tmp)
        else:
            for i in range(0, len(wordDict)):
                self._139_wordBreak_backtracking(output, tmp + wordDict[i], target, wordDict)

    def _141_hasCycle(self, head: ListNode) -> bool:
        if head == None:
            return False

        hashSet = set()

        current = head

        while current:
            if current in hashSet:
                return True

            hashSet.add(current)
            current = current.next

        return False

    def _142_detectCycle(self, head: ListNode) -> ListNode:
        if head == None:
            return False

        hashSet = set()

        current = head

        while current:
            if current in hashSet:
                return current

            hashSet.add(current)
            current = current.next

        return None

    def _143_reorderList(self, head: ListNode) -> None:
        if head == None or head.next == None:
            return

        slow = head
        fast = head.next.next

        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next

        middle = slow.next
        slow.next = None

        current = middle
        previous = None
        next = None

        while current != None:
            next = current.next

            current.next = previous

            previous = current
            current = next

        current = head
        middle = previous

        while current != None and current.next != None:
            tmp = current.next

            current.next = middle

            middle = middle.next
            current.next.next = tmp

            current = tmp

        current.next = middle

    def _144_preorderTraversal(self, root: TreeNode) -> List[int]:
        if root == None:
            return []

        values = []
        stack = []
        stack.append(root)

        while stack:
            tn = stack.pop()

            values.append(tn.val)

            if tn.right != None:
                stack.append(tn.right)
            if tn.left != None:
                stack.append(tn.left)

        return values

    def _145_postorderTraversal(self, root: TreeNode) -> List[int]:
        if root == None:
            return []

        values = []
        stack1 = []
        stack2 = []

        stack1.append(root)

        while stack1:
            tn = stack1.pop()
            stack2.append(tn)

            if tn.left != None:
                stack1.append(tn.left)
            if tn.right != None:
                stack1.append(tn.right)

        while stack2:
            values.append(stack2.pop().val)

        return values

    def _146_lruCache(self, size: int) -> LRUCache:
        return LRUCache(size)
