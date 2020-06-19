
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


# Multi-Pointer

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

# Index Manipulation

### Set Matrix Zeroes Leetcode 73 (medium)
https://leetcode.com/problems/set-matrix-zeroes/

* Note: in-place change

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

### Search a 2D Matrix Leetcode 74 (medium)
https://leetcode.com/problems/search-a-2d-matrix/

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


39. Combination Sum
33. Search in Rotated Sorted Array
81. Search in Rotated Sorted Array II
34. Find First and Last Position of Element in Sorted Array
48. Rotate Image
75. Sort Colors (sort, inplace, duplicate elements)
