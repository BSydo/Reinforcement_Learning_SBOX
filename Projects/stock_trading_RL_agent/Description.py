## Environment 
# mimics the OpenAI gym API
# state = env.reset(), next_state, reward, done, info = env.step(action)
# We'll consider 3 stocks: AAPL, MSI, SBUX

## state will consist of 3 parts:
# 1 - how many shares of each stock I own.
#   e.g. [3,5,7] means 3 shares of Apple, 5 of Motorola and 7 of Starbucks
# 2 - current price of each stock
#   e.g. [50,20,30] means Apple's stocks worth 50 each 
#   and 20 and 30 for Motorola and Starbucks
# 3 - Total cash we have currently
#   e.g. we have $100 cash, so my full vector is [3,5,7,50,20,30,100]
# If we have N stocks, then the state will contain 2N+1 components

## Actions (simplified):
# Many options to consider.
# For nany stock, I can buy/sell/hold.
# And in our world we have 3 stocks to consider.
# Thus we have 3^3 = 27 possibilites
#   e.g. [sell,sell,sell]
#   e.g. [buy.sell,buy]
 
## Simplicity:
# - ignore transactional costs
# - if we choose sell, we will sell all shares that we own
# - if we buy - we buy as many as possible. If we buy multiple stocks, we will 
#   do so in 'round robin' fashion - loop through every stock 
#   and buy 1 of each stock until we run out of money
# - Sell before buy
# One 'action' in our environment will involve performing all of these steps 
# at once.

## Reward:
# Change in value of portfolio from one step (state s) to the next (state s')
# How we will calculate value of portfolio:
#   Ex.: 
#       We own 10 shares of AAPL, 5 of MSI and 3 of SBUX
#       Share prices are [50,20,30]
#       Cash = $100
#       Total value of portfolio will be 10*50+5*20+3*30+100 = $790
# Reward is a change of portfolio value between steps
#   s = vector of number shares owned
#   p = vector of share prices
#   c = cash
#   portfolio value = s(transposed)*p+c
# Reward is the difference between portfolio value comparing the 
# most recent time step and the previous time step