Dynamic Programming



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


