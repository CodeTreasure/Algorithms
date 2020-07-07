content
# Introduction to Dynamic Programming (DP)
## What's a DP Problem?
Dynamic programming is a optimization method that is used to solve a complex problem by breaking the problem into easier subproblems and conquering them step by step.

The key features of DP problem are
* **Optimal Substructure**
* **Overlapping Sub-problems**

In short, if a problem could be solved by solution of some subproblems plus choosing a action from several choices, that's a DP problem.

* **Example 1: Fibonacci Sequence**. 
Fibonacci Sequence is a sequence starts from 0 and 1, the following nums always are the sum of two preceding ones. Fibonacci = [0, 1, 1, 2, 3, 5, 8, 13,...]

* **Example 2: Min Cost Climbing Stairs (Leetcode #746)**.
https://leetcode.com/problems/min-cost-climbing-stairs/. On a staircase, the i-th step has some non-negative cost cost[i] assigned. Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.


## Steps to Solve a DP Problem
The most difficult step to solve a DP problem is recognizing a DP problem. Once you recognize a DP problem, you've already solved half of the problem.

Next, you need to find:
* **States**: A state is a particular condition using a/several variable(s) to describe the situation at each step in the problem. In some simple problems, it may be the same as the result (min cost or something else). However, in some complicated problems, it may be combination of result and other constraints. For example, in some stock problems, the profit/cost and some limitations (can/cannot buy/sell stocks) describe your conditions every day. 
* **Actions**: Actions represents the acitons you can choose in step-by-step decision making process. For example, in the problem 'Min Cost Climbing Stairs', you can choose climbing 1 or 2 steps every time.
* **Transition**: Transition means the process that change one state to another state after taking some actions in each step. Still take Min Cost Climbing Stairs as an example, Transition means how you get result(n) from result(n-1), result(n-2)... Because you can only climb 1 or 2 steps, you just need to consider result(n-1)+cost[n-1] and result(n-2)+cost[n-2] and would know the less one of them is the min cost for n stairs. This part is the second difficult part in a DP problem. But you could master it after some practices.
* **Base Cases/States**: A base case is the start point of a problem as well as the most simple subproblem that doesn’t depend on other subproblems. For example, Fibonacci[0] = 0, Fibonacci[1] = 1.

After those steps, you've done 99% of a DP problem. Then, you need to choose how to store  solutions of sub-problems and solve the problem forward or backward:
* **Top-down Method/Memoization (Backward)**: This is a recursive method. You store the solution of subproblems in momery. For example, when you want Fibonacci(n), you just use Fibonacci(n-1)+Fibonacci(n-2), the program will automatically calculate Fibonacci(n-1), Fibonacci(n-2), Fibonacci(n-3)...backward until the base cases, after that, will return Fibonacci(n). This method takes a lot of memory because it calls the functions many times in stack and may not avaible for a large n.

* **Bottom-up Method/Tabulation (Forward)**: This method runs faster than the previous one. You can start from the base case and use a dp table to store solutions of subproblems. Then, derive the next step from previous solution. In some cases, you can store last several solutions rather than all previous solutions to reduce memory usage from O(n) to O(1).

The following code illustrates the difference of thoese two methods.
```python3
def Fibonacci_top_down(n):
    if n<0: return "N should be a positive int"
    elif int(n)==0: return 0
    elif int(n)==1: return 1
    return Fibonacci_top_down(n-1)+Fibonacci_top_down(n-2)
    
 
def Fibonacci_bottom_up(n):
    if n<0: return "N should be a positive int"
    elif int(n)==0: return 0
    res = [0, 1]
    for i in range(int(n-1)):
        res.append(res[-1]+res[-2])
    return res[-1]
    
    
def Fibonacci_bottom_up_improvement(n):
    if n<0: return "N should be a positive int"
    elif int(n)==0: return 0
    previous, current =0, 1
    for i in range(int(n-1)):
        previous, current = current, current+previous
    return current
```

Cheers! Now, you are a DP expert who knows everything about DP :)

Whoops, you find a DP problem **TOOOOO HARD** to deal with? 

Let's start with some practices and you will conquer any DP problems after that. (I promise!!! If you won't, just come back and **PRACTICE MORE (^_^) **.

# Learn DP by Solving Leetcode Problems


### Simple Example: Climbing Stairs (Leetcode #70)
https://leetcode.com/problems/climbing-stairs/

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Obviously, 
* n = 1, ans = 1
* n = 2, ans = 2 (1+1 or 2)
* n = 3, ans = ans(1) + ans(2) (climb 2 steps from 1 or climb 1 step from 2)
* n = ... ans = ans(n-2) + ans(n-1)

```python3
def climbStairs(self, n: int) -> int:
        if n == 1 : return 1
        cur = 2
        lag = 1
        for i in range(2,n):
            lag, cur = cur, cur + lag
        return cur
```

### Min Cost Climbing Stairs Leetcode #746 
https://leetcode.com/problems/min-cost-climbing-stairs/

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.


```python3
def minCostClimbingStairs(self, cost: List[int]) -> int:
        lag_cost = 0
        cur_cost = 0
        lag_c = cost[0]
        cur_c = cost[1]
        
        for c in cost[2:]:
            lag_cost, cur_cost = cur_cost, min(lag_cost+lag_c, cur_cost+cur_c)
            lag_c, cur_c = cur_c, c
        return min(lag_cost+lag_c, cur_cost+cur_c)
```
## Classical Problems
### Coin Change (Leetcode #322, Medium)
https://leetcode.com/problems/coin-change/

```python3
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int: 
        # 50% 1480
        dp = [0]+[amount+10]*amount
        for i in range(1, amount+1):
            for coin in coins:
                dp[i] = min(dp[i], dp[i-coin]+1) if i-coin>=0 else dp[i]            
        return dp[-1] if dp[-1]!=amount+10 else -1
```

## Subarray Problems (Maximum Sum, Maximum Product...)
### Maximum Subarray Leetcode #53
https://leetcode.com/problems/maximum-subarray/

* **Problem**: 
* **Actions**: [previous + current_value, starting with current_value]
* **States**: the maximum at each step

```python3
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [float('-inf')]
        for num in nums:
            dp.append(max(dp[-1]+num, num))
        return max(dp)
```
```python3
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curr_max = nums[0]
        global_max = nums[0]
        for i in range(1, len(nums)):
            curr_max = max(nums[i], curr_max+nums[i])
            global_max = max(global_max, curr_max)
        return global_max
```

### Maximum Product Subarray #152
https://leetcode.com/problems/maximum-product-subarray/

* **Problem**: Given a list nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

* **States**: 
* **Actions**: [previous * current_value, starting with current_value]
* **Transition**: 
* **Base Case**: dp[0]=nums[0]


### Longest Increasing Subsequence (Leetcode #300)
* **Problem**: Given an unsorted array of integers, find the length of longest increasing subsequence.
* **Note**: Subsequence may be not countinous in the original array.
* **Example**: [1,3,6,7,9,4,10,5,6] -> 6 ([1,3,6,7,9,10])
* **States**: In this problem, every state means the length of increasing subsequence to the index, the solution is max(dp). dont use dp[n] to represent the solution, since we need another parameter to record the position, at which we get the LIS. 
* **Actions**: we need to compare nums[i], nums[j] for any j<i and dp[j] to decide the value of dp[i]
* **Transition**: if nums[j]<nums[i], we can inherit the dp[j] otherwise we cannot get useful information from dp[j] since we dont know how many nums before nums[j] should be included. Therefore, for dp[i], we need an array temp to represent the inheritance from dp[j]. if nums[j]<nums[i], temp[j] = dp[j]+1, elsewise 0. dp[i] = max(temp)
* **Base Case**: The innital state could be the 1s.

```python3
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0
        dp = [1]*len(nums)
        for i in range (1, len(nums)):
            for j in range(i):
                if nums[i] >nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```





## Word Problems(, Break, Split...)

### Longest Substring Without Repeating Characters (Leetcode #3, Medium)
https://leetcode.com/problems/longest-substring-without-repeating-characters/

* **States**: dp[i] represents the length of substring without repreating characters ending with index i
* **Actions**: For each step s[i], we need to decide if add it to current substring s[a:i] or start with a char behind s[a], eg s[b:i]+s[i], b>a
* **Transition**: if s[i] appears in s[a:i], we can need to find b>a that s[a:i]+s[i] dont have repeating chars. Otherwise, we just add s[i] to s[a:i], and dp[i]=dp[i-1]+1. The only difficult thing is the starting position of each dp[j] is different, if we s[i] appears in s[a:i], we need to find the starting point of dp[i]. so the length should be min of dp[j]+i-j for all j<i. 
* **Base Case**: 1's for all individual chars


```python3
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: return 0
        dp = [1]*len(s)
        
        for i in range(1, len(s)):
            local_length = float("inf")
            for j in range(i-1, -1, -1):
                if s[i]==s[j]:
                    break
                else:
                    dp[i]+=1
                    local_length = min(dp[j]+i-j, local_length)
            
            dp[i] = min(dp[i], local_length)            
        return max(dp)
```

* **Improvement**: The previous complexity is O(n^2). Actually, we could know the starting index for the substring ending with i. 

```python3
    def lengthOfLongestSubstring(self, s: str) -> int:
        record = {}
        curr_string = 0
        res = 0
        for i in range(len(s)):
            if s[i] not in record:
                curr_string = curr_string+1
            else:
                curr_string = min(curr_string+1, i - record[s[i]])
                
            record[s[i]] = i
            res = max(res, curr_string)
            
        return res

```


### Word Break (Leetcode #139, Medium)
https://leetcode.com/problems/word-break/

* **Problem**: Given a string s and a wordDict: List[str] . To test if the string could be separated to combination of words in the word list.
* **Note**: the same word in the word list can be used many times.

* **States**: Every index represents a step means if s[:idx] can be broken into words in wordDict.
* **Actions**: For each state dp[j], you can choose a few successive i-j chars as a action and move to dp[i]
* **Transition**: Obviously, if dp[j] is True and s[j:i] is a word in wordDict for any j < i, then dp[i] is True. Otherwise, dp[i] is False
* **Base Case**: If we dont specify the case s[0], we need to make the 1st element in dp table True.
* **Imporvement**: For most cases, len(word) is much smaller than len(s), so updating backward from dp[i-1] with s[i-1:i] is faster than forward from dp[0] with s[:i].

```python3
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # 92%
        dp = [True]+[False]*len(s)
        for i in range(len(dp)):
            # for j in range(i):
            for j in range(i-1, -1, -1):
                if dp[j] and s[j:i] in wordDict: 
                    dp[i]=True
                    break
        return dp[-1]
```

### Word Break 2 (Leetcode #140, Hard)
https://leetcode.com/problems/word-break-ii/

* **Change**: Return all such possible combination of words in wordDict rather than just returning True or False.
* **Example**: s = "catsanddog",
wordDict = ["cat", "cats", "and", "sand", "dog"] -> 
[
  "cats and dog",
  "cat sand dog"
]

* **States**: In this problem, every state need to record the sentence (combination of words) for the corresponding index.
* **Actions**: For each state dp[j], you can choose a few successive i-j chars as a action and move to dp[i]
* **Transition**: Obviously, if dp[j] has some sentences and s[j:i] is a word in the wordDict for some j < i, then dp[i] equal to every sentence in dp[j] + " "+ s[j:i]. Otherwise, dp[i] is []
* **Base Case**: If we dont specify the case s[0], we need to make the 1st element in dp table [""] for iteration later.
* **Special Cases**: For most cases, if the number of different chars in s is larger than that in wordDict, we know s cannot be separated into words in the wordDict 

```python3
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # 95% 36ms
        if set(collections.Counter(s).keys()) > set(collections.Counter("".join(wordDict)).keys()):
            return []
        
        length = len(s)
        dp = {0: [""]}
        for i in range(length+1):
            for j in range(i):
                if j in dp and s[j:i] in wordDict:
                    if i not in dp:
                        dp[i]=[words+' '+s[j:i] for words in dp[j]]
                    else:
                        dp[i]+=[words+' '+s[j:i] for words in dp[j]]
        # sentence[1:] is faster than sentence.strip()
        return [sentence[1:] for sentence in dp.get(length, [])]
```
### Next 472. Concatenated Words


### Paint House Leetcode #256 
https://leetcode.com/problems/paint-house/

Paint a row of hourses, costs for painting different houses are different. 
n * 3 means cost of [red, blue, green] for n houses and no two adjacent houses have the same color
* **Example**: [[17,2,17],[16,16,5],[14,3,19]] -> 10 (blue, green, blue)

Obviously, it is the DP problem. In each step, we just update the record of the min total cost of painting 1->ith house
* **Actions**: 
[red, blue, green]
* red = min(cost_red+previous_blue, cost_red+previous_green)
* blue = min(cost_blue+previous_red, cost_blue+previous_green)
* green = min(cost_green+previous_red, cost_green+previous_blue)

```python3
def minCost(self, costs: List[List[int]]) -> int:
        red, blue, green = 0,0,0
        for cost in costs:
            red, blue, green = cost[0]+min(blue, green), cost[1]+min(red, green), cost[2]+min(red, blue)
        return min(red, blue, green)
```

### Paint House II Leetcode #265 Hard
https://leetcode.com/problems/paint-house-ii/

* **Changes**: 3 -> k colors. Similar to the previous one. 
Just change the size of the record from 3 to k

```Python
def minCostII(self, costs: List[List[int]]) -> int:
        if not costs: return 0
        if len(costs)==1: return min(costs[0])
        previous_rec = [0]*len(costs[0])
        current_rec = [0]*len(costs[0])
        for cost in costs:
            for i in range(len(cost)):
                current_rec[i] = cost[i]+min(previous_rec[:i]+previous_rec[i+1:])
            previous_rec = current_rec.copy()
        return min(current_rec)
```



### House Robber Leetcode #198 
https://leetcode.com/problems/house-robber/

Given an array of money in every house, U cannot rob two adjacent houses, find the max money you can rob.

* [1,2,3,1] -> 4
* [2,7,9,3,1] -> 12
* [2,1,1,2] -> 4

Two actions: rob(r) or not(n)

In each step (for money in moneys)
* rob = previous_not + money
* not = max(previous_not, previous_rob)

Finally, just return max(rob, not). 

```python3
def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        r = nums[0]
        n = 0
        for num in nums[1:]:
            r, n = n+num, max(r, n)
        return max(r, n)
```

### House Robber II Leetcode #213 
https://leetcode.com/problems/house-robber-ii/

Changes: houres arranged in a circle, which means you cannot rob the 1st one and the last one together

We can reuse our previous solution and make a tiny change: 
* if 1st house not rob, just ignore the first one (nums[1:])
* if 1st house robbed, just cal the rest (nums[2:-1]) + nums[0]

```python3
def rob_2(nums)
        if not nums: return 0
        return max(rob(nums[1:]), nums[0]+rob(nums[2:-1]))  
        
def rob_1(nums): 
        if not nums: return 0
        r = nums[0]
        n = 0
        for num in nums[1:]:
            r, n = n+num, max(r, n)
        return max(r, n)
```


## 2D-array Problems ()

### Unique Paths Leetcode #62
https://leetcode.com/problems/unique-paths/

* **Problem**: a robot starts from the top-left cell of a m\*n grid (n rows, m cols) and can move down or right each step. how many paths for the robot to the bottom-right cell?
* **Actions**: [right, down]
* **Initial States**: elemnts in the 1st row and the 1st col must be 1, can only reach by right->right->...->right or down->down->...->down

```python3
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*m for i in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[n-1][m-1]
```

### Unique Paths II Leetcode #63
https://leetcode.com/problems/unique-paths-ii/

* **Change**: there are some obstacles in the grids (marked as 1)
* **Example**: [[0,0,0],[0,1,0],[0,0,0]] -> 2
* **Actions**: [right, down] same as the previous problem
* **Initial States**: similar to preivous problem, but need to mark obstacles to 0

```python3
class Solution:
    # faster than 99.9%
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0] == 1: return 0
        dp = [[1-element for element in row] for row in obstacleGrid]
        n, m = len(obstacleGrid), len(obstacleGrid[0])
        for i in range(n):
            for j in range(m):
                if i == 0 and j>0: 
                    dp[i][j] = dp[i][j-1] if dp[i][j]!=0 else 0
                elif j == 0 and i>0:
                    dp[i][j] = dp[i-1][j] if dp[i][j]!=0 else 0
                elif i>0 and j>0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1] if dp[i][j]!=0 else 0
        return dp[n-1][m-1]
```

### Minimum Path Sum  Leetcode #64
* **Problem**: similar to the Unique Paths (Leetcode #62), but every cell has a cost, need to find the min cost path to the bottom-right cell.

```python3
class Solution:
    # faster than 98%
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i==0:
                    if j>0: grid[i][j]+=grid[i][j-1]
                elif j==0:
                    grid[i][j]+=grid[i-1][j]
                else:
                    grid[i][j]+=min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]
```


### Leetcode #120 Triangle (Medium)
* **Problem**: Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
* **Top-Bottom**: if we know the total cost(sum) of each element of the row above, we could calculate the total cost of this row. (just add up the element with adjacent cost in the row above and find the less one)
* **Note**: for the 1st/last element of each row, we can only use 1st/last cost

```python3
def minimumTotal(self, triangle: List[List[int]]) -> int:
        res = [0]
        for row in triangle:
            temp = []
            temp.append(res[0]+row[0])
            for i in range(1, len(row)):
                temp.append(min(res[i-1], res[i])+row[i])
            temp.append(res[-1]+row[-1])
            res = temp
        return min(res)
```

### Range Sum Query 1D - Immutable (Leetcode #303) 
https://leetcode.com/problems/range-sum-query-immutable/

Not so close to DP, but follow-up questions are typical dp questions.
```python3
class NumArray:
    def __init__(self, nums: List[int]):
        self.sums = nums
        for i in range(1, len(nums)):
            self.sums[i]+=self.sums[i-1]

        
    def sumRange(self, i: int, j: int) -> int:
        return self.sums[j] if i==0 else self.sums[j]-self.sums[i-1]
```

### Range Sum Query 2D - Immutable (Leetcode #304, Medium) 


```python3
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        if not matrix: return None
        self.sums =[[0]+row for row in matrix]
        self.sums = [[0]*len(self.sums[0])] + self.sums
        for i in range(1, len(self.sums)):
            for j in range(1, len(self.sums[i])):
                self.sums[i][j] += self.sums[i][j-1]
            for j in range(1, len(self.sums[i])):
                self.sums[i][j] += self.sums[i-1][j]
        
        
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = self.sums[row2+1][col2+1] - self.sums[row1][col2+1] - self.sums[row2+1][col1] + self.sums[row1][col1]
        return(res)
        
```

### Range Sum Query 2D - Mutable (Leetcode #308, Hard) 


similar to previous one, we have 

```python3
class NumMatrix:
    # 20% 1500+
    def __init__(self, matrix: List[List[int]]):
        if not matrix: return None
        self.matrix = matrix
        self.sums =[[0]+row for row in matrix]
        self.sums = [[0]*len(self.sums[0])] + self.sums
        for i in range(1, len(self.sums)):
            for j in range(1, len(self.sums[i])):
                self.sums[i][j] += self.sums[i][j-1]
            for j in range(1, len(self.sums[i])):
                self.sums[i][j] += self.sums[i-1][j]

                
    def update(self, row: int, col: int, val: int) -> None:
        # update sums
        for i in range(row+1, len(self.sums)):
            for j in range(col+1, len(self.sums[0])):
                self.sums[i][j] += val-self.matrix[row][col]
        
        # update matrix
        self.matrix[row][col] = val
                
                
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = self.sums[row2+1][col2+1] - self.sums[row1][col2+1] - self.sums[row2+1][col1] + self.sums[row1][col1]
        return(res)
```

* **Improvement**: (however, if we use this in previous problem, it becomes slower.)

```python3
class NumMatrix:
    # 68% 164
    def __init__(self, matrix: List[List[int]]):
        if not matrix: return None
        self.matrix = matrix
        self.sums =[[0]+row for row in matrix]
        self.sums = [[0]*len(self.sums[0])] + self.sums
        for i in range(1, len(self.sums)):
            for j in range(1, len(self.sums[i])):
                self.sums[i][j] += self.sums[i][j-1]
                
    def update(self, row: int, col: int, val: int) -> None:
        # update sums
        for j in range(col+1, len(self.sums[0])):
            self.sums[row+1][j] += val-self.matrix[row][col]
        
        # update matrix
        self.matrix[row][col] = val
                
                
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = 0
        for i in range(row1+1, row2+2):
            res += self.sums[i][col2+1] - self.sums[i][col1]
        return(res)
```


### Maximal Square (Leetcode #221) 
https://leetcode.com/problems/maximal-square/

* **Problem**: Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
* **Example**: [[0 0 1],[0 1 1],[1 1 1]] -> 4
* **States**: In this problem, every state need to record the square/sides for the corresponding positions.
* **Actions**: For each state dp[i,j], we know the adjacent elements will impact the next states
* **Transition**: if dp[i-1,j-1], dp[i-1,j], dp[i,j-1] has the same side n and matrix[i,j] == 1, then dp[i,j] = n+1. otherwise, then dp[i,j] is decided by the min side of dp[i-1,j-1], dp[i-1,j] and dp[i,j-1]. It's easy to plot a figure to proof that.
* **Base Case**: The innital state could be the same as the input or 0s.

```python3
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        # 80%
        if matrix is None or len(matrix) < 1:
            return 0
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        dp = [[0]*(cols+1) for _ in range(rows+1)]
        max_side = 0
        
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == '1':
                    dp[r+1][c+1] = min(dp[r][c], dp[r+1][c], dp[r][c+1]) + 1 
                    max_side = max(max_side, dp[r+1][c+1])
                
        return max_side * max_side
 ```
 



# Reference
* https://en.wikipedia.org/wiki/Dynamic_programming
* https://www.geeksforgeeks.org/solve-dynamic-programming-problem/
