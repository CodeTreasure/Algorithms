The 1st essay on Github
## Problem Introduction
This essay explains how to use DP (Dynamic Programming) to solve stock problems on Leetcode.

This series of problems ask you the max profit you can get if you know the stock price in the next few days with some restrictions.

Let's see the simplest stock problem: Leetcode #121

Best Time to Buy and Sell Stock
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Input: 
prices(List[int]) the stock price on day i
Output: 
MaxProfit(int) the max profit you could get
1. Only one transaction (buy and sell once)
2. 0 <= positions <= 1 (You cannot short stock and hold more than 1 share)


## Why Dynamic Programming
In this problem, we have len(prices) steps and have 3 different actions and 2 states in each step.

actions: buy, hold, sell
states: money(cash on your account) and position(how many shares do you currently hold).

Therefore, it is not hard for us to come up with DP solutions.

### How to Slove it Mathematically
we use n represents the target function of the state(position=0), b represents the target function of the state(position=1) and we update states each day.

if position=0: that means two possible cases.
1: Yesterday position = 0 and we don't buy it today.
2: Yesterday position = 1 but we sell it today.

Similarly, 
if position=1: that also have two possible cases.
1: Yesterday position = 1 and we don't sell it today.
2: Yesterday position = 0 but we buy it today.

Finally, we just need to return n since you must sell the stock to get the max profit (given prices>=0).

And at the beginning, n = 0 and b = -infinity (since it is unavaible for you get profit on Day 1), after that, just updates those states in each step.

### Answer
```python3

def maxProfit(self, prices: List[int]) -> int:
# 只交易一次，所以我们找到之前价格最低点，算出来每一步卖出的利润，最后看利润的最大值就可以
if len(prices)<2: return 0

# 初始化第一个值
lowest_price = [prices[0]]
res = [0]

# 对于每个price，和之前的最低值比较，然后算当前价格-最低买入价格的利润
for price in prices[1:]:
    lowest_price.append(min(price, lowest_price[-1]))
    res.append(price-lowest_price[-1])

return max(res)


def maxProfit(self, prices: List[int]) -> int:
        # DP
        if len(prices)<2:
            return 0
        profit = 0
        cost = -prices[0]
        
        for p in prices:
            profit = max(profit, cost+p)
            cost = max(-p, cost)
            # print(cost, profit)
            
        return profit
```

Leetcode #122
Best Time to Buy and Sell Stock II
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

1. No limitation on the number of transaction. However, you cannot hold >1 stocks (position<=1)
2. Cannot short stock (position >=0)

```python3

def maxProfit(self, prices: List[int]) -> int:
    # DP
    # 在每天用两个值分别统计，没有股票，以及购买股票的profit最大值
    # 每一天，没有股票的情况，等于 max(上一天有股票卖出, 上一天没有股票利润)
    # 有股票，等于 max(上一天没有股票但今天买了股票, 上一天有股票但是今天没卖的利润)
    if not prices:
        return 0

    n = len(prices)
    dp = [[0] * 2 for _ in range(n)]
    
    # 初始化第一天的值
    dp[0][0] = 0  # No stock on day 0, profit = 0
    dp[0][1] = -prices[0]  # Bought stock on day 0, negative profit
    
    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
        dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])

    return dp[-1][0]  # Maximum profit at last day without holding stock


def maxProfit2(self, prices: List[int]) -> int:
        # DP 
        if len(prices)<2: return 0
        profit = 0
        cost = -prices[0]
        
        for p in prices:
            profit = max(profit, cost+p)
            cost = max(profit-p, cost)
            
        return profit
```


Leetcode #123
Best Time to Buy and Sell Stock III
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
1. transactions<=2 (you can have at most 2 transactions). 
2. 0 <= positions <= 1 (You cannot short stock and hold more than 1 share)

```python3
def maxProfit3(self, prices: List[int]) -> int:
            b_1 = -prices[0]
            n_1 = 0
            b_0 = -999
            n_0 = 0
            for price in prices:
                n_0 = max(n_0, b_0+price)
                b_0 = max(b_0, n_1-price)
                n_1 = max(n_1, b_1+price)
                b_1 = max(b_1, n_2-price)
            return n_0
```


Leetcode #188
Best Time to Buy and Sell Stock IV
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/

1. transactions<=k (you can have at most k transactions). 
2. 0 <= positions <= 1 (You cannot short stock and hold more than 1 share)

```python3
    def maxProfit4(self, k: int, prices: List[int]) -> int:
        if len(prices)<2:
            return 0
        else:
            n = [0]*k
            b = [-9999]*k
            for price in prices:
                for i in range(k):
                    b[i] = max(n[i-1]-price, b[i])
                    n[i] = max(n[i], b[i]+price)
            return n[-1]
```


Leetcode #309
Best Time to Buy and Sell Stock with Cooldown
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

1. No limitation on the number of transaction. However, you cannot hold >1 stocks (position<=1)
2. Cooldown:1 day (if you sell it on i th day, you cannot buy it on i+1 th day.

```python3
def maxProfit_cooldown(self, prices: List[int]) -> int:
        # DP
        if len(prices) >2:
            n = 0
            b = -prices[0]
            lag1_n = 0
            for p in prices:
                lag2_n = lag1_n
                lag1_n = n
                n = max(n, b+p)
                b = max(b, lag2_n - p)
            return n
        
        elif len(prices) < 2: return 0
        else: return max(0, prices[1]-prices[0])
```


Leetcode #714
Best Time to Buy and Sell Stock with Transaction Fee
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/

1. No limitation on the number of transaction. However, you cannot hold >1 stocks (position<=1)
2. A non-negative fee on each transaction(i.e. buying at 1 and selling at 7, the profit is (7-1)-fee)

0 < prices.length <= 50000.

```python3
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = 0
        b = -prices[0]
        for p in prices:
            n = max(n, b+p-fee)
            b = max(b, n-p)
        return n
```

1st draft
