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


### Paint House Leetcode #256 
https://leetcode.com/problems/paint-house/

Paint a row of hourses, costs for painting different houses are different. 
n * 3 means cost of [red, blue, green] for n houses and no two adjacent houses have the same color
* [[17,2,17],[16,16,5],[14,3,19]] -> 10 (blue, green, blue)

Obviously, it is the DP problem. In each step, we just update the record of the min total cost of painting 1->ith house
* actions = [red, blue, green]
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

Changes: 3 -> k colors. Similar to the previous one. 
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



