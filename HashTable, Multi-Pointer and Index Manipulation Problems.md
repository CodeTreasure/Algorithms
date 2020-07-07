
# Hash Table

### Simple Example: Two Sum (Leetcode #1)
https://leetcode.com/problems/two-sum/


```python3
def twoSum(self, nums: List[int], target: int) -> List[int]:
        rec = {}
        rec[nums[0]] = 0
        for i in range(1, len(nums)):
            if (target - nums[i]) in rec:
                return rec[target - nums[i]], i
            else: rec[nums[i]] = i
```

### Two Sum II - Input array is sorted
https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

```python3
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i = 0
        j = len(numbers)-1
        while i<j:
            if numbers[i] + numbers[j] == target:
                return [i+1,j+1]
            elif numbers[i] + numbers[j] > target:
                j-= 1
            else:
                i += 1
        return []
```

### Two Sum III - Data structure design Leetcode #170

```python3
# dict 90%
class TwoSum:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums = {}

    def add(self, number: int) -> None:
        """
        Add the number to an internal data structure..
        """
        self.nums[number] = self.nums.get(number, 0) + 1

    def find(self, value: int) -> bool:
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        """
        for key in self.nums.keys():
            residual = value-key
            if residual in self.nums:
                if residual!=key or (residual==key and self.nums[key]>1): return True         
        return False
```


## Multi-Pointer

### Two Sum III - Data structure design Leetcode #170


```python3
# List 10%
class TwoSum:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums = []
    
    def add(self, number: int) -> None:
        """
        Add the number to an internal data structure..
        """
        self.nums.append(number)

    def find(self, value: int) -> bool:
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        """
        if len(self.nums)<2 : return False
        
        temp = self.nums.copy()
        temp.sort()
        l, r = 0, len(temp)-1
        while l<r:
            if temp[l]+temp[r]<value:
                l+=1
            elif temp[l]+temp[r]>value:
                r-=1
            else:
                return True
        return False
```



### 3 Sum Leetcode #198
https://leetcode.com/problems/3sum/

Find unique triplets in a given array that sum of every triplet = 0
* [-1, 0, 1, 2, -1, -4] -> [[-1, 0, 1],[-1, -1, 2]]

Time Limit Exceeded
```python3
def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        length = len(nums)
        res = []
        for i in range(length-2):
            if nums[i]>0: break
            if i>0 and nums[i]==nums[i-1]: continue
        
            l, r = i+1, length-1
            while l<r:
                if nums[l]+nums[r]+nums[i]<0:
                    l+=1
                elif nums[l]+nums[r]+nums[i]>0:
                    r-=1
                else:
                    if [nums[i], nums[l], nums[r]] not in res:
                        res.append([nums[i], nums[l], nums[r]])
                    l+=1
                    r-=1

```

### Minimum Size Subarray Sum (Leetcode #209 Medium)
https://leetcode.com/problems/minimum-size-subarray-sum/

* **Problem**: Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.


```python3
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        # 2 pointers
        if sum(nums)>=s: res = len(nums)
        else: return 0
        cur_sum = 0
        start_idx, end_idx = 0, 0
        while start_idx<len(nums) and end_idx<len(nums):
            cur_sum += nums[end_idx]
            while cur_sum>=s:
                res = min(res, end_idx-start_idx+1)
                cur_sum -= nums[start_idx]
                start_idx+=1
                
            end_idx+=1
        return res
```




### Container With Most Water (Leetcode #11)
https://leetcode.com/problems/container-with-most-water/

This is a two pointer problem. The key idea is when to move your points. 
You cannot move ...

When you've already had current area, how to make is important

At first, we know area = lower_bar * distance

Just imagine, when the left bar is lower than the right one, we know the area is bounded by the left bar: area = left_bar * distance. 

if you move the right bar to the left, you can never get a larger area, since new area = left_bar * (distance-positive_integer)

Therefore, we need to move the lower bar each time.

```python3
# 90%
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height)-1
        res = 0
        while l<r:
            current = min(height[l], height[r])*(r-l)
            res = max(res, current)
            if height[r]<height[l]:
                r-=1
            else:
                l+=1
        return res
```

## Bidirection Subarrays

### Product of Array Except Self (Leetcode #238 Medium)
https://leetcode.com/problems/product-of-array-except-self/

* **Problem**: Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
* **Note**: Solve without division and in O(n)

```python3
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res, left_prod, right_prod = [1]*len(nums), [1]*len(nums), [1]*len(nums)
        for i in range(1, len(nums)):
            left_prod[i] = left_prod[i-1]*nums[i-1]
            right_prod[-i-1] = right_prod[-i]*nums[-i]
            res[i] *= left_prod[i]
            res[-i-1]*= right_prod[-i-1]
        return res
```

### Trapping Rain Water (Leetcode #42 Hard)
https://leetcode.com/problems/trapping-rain-water/

* **Idea**: for each step[i], the volume of water is equal to min(left_max, right_max)-height[i]. ***left_max*** is the maximum height of bar from the left end upto the index i, so is ***left_right***

```python3
left_max, right_max = 0, 0
        res = [float("inf")]*len(height)
        for i in range(len(height)):
            left_max=max(left_max, height[i])
            res[i] = min(res[i], left_max-height[i])
            right_max=max(right_max, height[-i-1])
            res[-i-1] = min(res[-i-1], right_max-height[-i-1])
        return sum(res)
```






# Index Manipulation

## Matrix

### Set Matrix Zeroes (Leetcode #73 Medium)
https://leetcode.com/problems/set-matrix-zeroes/

* **Problem**: Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.
* **Note**: in-place change

Need to improve

```python3
class Solution:
    # 50%
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rec = [[i,j] for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] ==0]
        rows = [idx[0] for idx in rec]
        cols = [idx[1] for idx in rec]
        for i in range(len(matrix)):
            if i in rows:
                matrix[i] = [0]*len(matrix[0])
            else:
                for j in range(len(matrix[0])):
                    if j in cols:
                        matrix[i][j] = 0
```

### Search a 2D Matrix (Leetcode #74 Medium)
https://leetcode.com/problems/search-a-2d-matrix/

* **Problem**: Given a m x n sorted matrix, Integers in each row are sorted from left to right. The first integer of each row is greater than the last integer of the previous row. Find if a number is in the Matrix.

* binary search
```python3
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix: return False
        n, m = len(matrix), len(matrix[0])
        l, r = 0, n*m-1
        while l<=r:
            mid = (l+r)//2
            if matrix[mid//m][mid%m] > target:
                r = mid-1
            elif matrix[mid//m][mid%m] < target:
                l = mid+1
            else:
                return True
        return False
```

* Idea: since we know it is sorted, we can compare the 1st and the last element of each row and find the row where the value is located.

```python3
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 90%
        if not matrix or not matrix[0]: return False
        start = [i for i in range(len(matrix)) if matrix[i][0]<=target]
        end = [i for i in range(len(matrix)) if matrix[i][-1]>=target]
        if not start or not end: return False
        if start[-1] == end[0]:
            for v in matrix[end[0]]:
                if v == target: return True
            return False
        else: return False
```

* Improve: use binary search when we find that row.
```python3 
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 97%
        if not matrix or not matrix[0]: return False
        start = [i for i in range(len(matrix)) if matrix[i][0]<=target]
        end = [i for i in range(len(matrix)) if matrix[i][-1]>=target]
        if not start or not end: return False
        if start[-1] == end[0]:
            row = end[0]
            l, r = 0, len(matrix[0])-1
            while l<=r:
                mid = (l+r)//2
                if matrix[row][mid] > target:
                    r = mid-1
                elif matrix[row][mid] < target:
                    l = mid+1
                else:
                    return True
            return False
        else: return False
```


### Game of Life (Leetcode #289 Medium)
https://leetcode.com/problems/search-a-2d-matrix/

* **Problem**: Given a m x n cells, each cell has initial state live(1) or dead(0). they can update their status with eight neighbors.
* **Transition 1**: Any live cell with fewer than (<) 2 or more than (>) 3 live neighbors dies. keeps live with 2 or 3 live cells. 
* **Transition 2**: Any dead cells with (==) 3 live neighbors becomes a live cell.

```python3
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        90%
        count = [[0]*len(row) for row in board]
        new = [[0] + row + [0] for row in board]
        new.insert(0, [0]*len(new[0]))
        new.append([0]*len(new[0]))
        
        for i in range(len(count)):
            for j in range(len(count[i])):
                count[i][j] = new[i][j]+new[i][j+1]+new[i][j+2]+new[i+1][j]+new[i+1][j+2] + new[i+2][j]+new[i+2][j+1]+new[i+2][j+2]
                
        for i in range(len(board)):
            for j in range(len(board[i])): 
                if board[i][j] == 0:
                    board[i][j] = 1 if count[i][j] == 3 else 0
                elif count[i][j] < 2 or count[i][j] >3:
                        board[i][j] = 0

```
* **Improvement**: Use dummy variables to record preivous cell.
```python3
code
```



39. Combination Sum
33. Search in Rotated Sorted Array
81. Search in Rotated Sorted Array II
34. Find First and Last Position of Element in Sorted Array
48. Rotate Image
75. Sort Colors (sort, inplace, duplicate elements)
