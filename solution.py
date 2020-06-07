import sys
from collections import deque
from typing import List

from common import ListNode
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

    def _7_reverse(self, x: int) -> int:
        min_limit = -0x80000000  # hex(-2**31-1)
        max_limit = 0x7fffffff  # hex(2**31-1)

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


def main():
    pass


if __name__ == "__main__":
    main()
