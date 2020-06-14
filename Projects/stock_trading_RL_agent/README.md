## How to run it:
for train:
python Stock_trader_RL.py -m train && plot_results.py -m train

for test:
python Stock_trader_RL.py -m test && plot_results.py -m test

## Environment 
mimics the OpenAI gym API

state = env.reset(), next_state, reward, done, info = env.step(action)
Well consider 3 stocks: AAPL, MSI, SBUX
state = [# shares owned, share prices, cash]
Actions = [buy, sell, hold]
Reward = change in the value of our portfolio
 
Its going to accept a time-series of stock prices as input into its 
construstor. Well also will have a pointer to tell us what day it is so we 
know the current stock prices. How much cash we initially start with is
initial_investment. 
Pseudocode:

class Environment:
  def __init__(self, stock_prices, initial_investment):
      self.pointer = 0
      self.stock_prices = stock_prices
      self.initial_investment = initial_investment
  def reset(self):
      # reset pointer to 0 and return initial state
  def step(self, action):
      # perform the trade, move pointer
      # calculate reward, next state, portfolio value, done


## State 
Will consist of 3 parts:
1 - how many shares of each stock I own.
  e.g. [3,5,7] means 3 shares of Apple, 5 of Motorola and 7 of Starbucks
2 - current price of each stock
  e.g. [50,20,30] means Apples stocks worth 50 each 
  and 20 and 30 for Motorola and Starbucks
3 - Total cash we have currently
  e.g. we have $100 cash, so my full vector is [3,5,7,50,20,30,100]
If we have N stocks, then the state will contain 2N+1 components

## Actions (simplified):
Many options to consider.
For nany stock, I can buy/sell/hold.
And in our world we have 3 stocks to consider.
Thus we have 3^3 = 27 possibilites
  e.g. [sell,sell,sell]
  e.g. [buy.sell,buy]

## Reward:
Change in value of portfolio from one step (state s) to the next (state s)
How we will calculate value of portfolio:
  Ex.: 
      We own 10 shares of AAPL, 5 of MSI and 3 of SBUX
      Share prices are [50,20,30]
      Cash = $100
      Total value of portfolio will be 10*50+5*20+3*30+100 = $790
Reward is a change of portfolio value between steps
  s = vector of number shares owned
  p = vector of share prices
  c = cash
  portfolio value = s(transposed)*p+c
Reward is the difference between portfolio value comparing the 
most recent time step and the previous time step
 
## Simplicity:
- ignore transactional costs
- if we choose sell, we will sell all shares that we own
- if we buy - we buy as many as possible. If we buy multiple stocks, we will 
  do so in round robin fashion - loop through every stock 
  and buy 1 of each stock until we run out of money
- Sell before buy
One action in our environment will involve performing all of these steps 
at once.

## Model
Q-Learning
Modelling Q(s,a) with linear regression
But try to treat the model more like how its done in Deep Reinforrcement 
Learning. Instead of transforming (s,a) into a feature vector x, we will 
use only the state and We will have a separate output for each action.

Q(s,:) = W(transposed)*s+b

In our case:
  State size = 7 (# of shares owned, prices, cash)
  Action size 3^3 = 27
  W is of size 7x27, b is of size 27
 
In order to update our model with Q-Learning, we are going to treat it like 
a supervised learning problem and do one step of Gradient Descentfor for 
each (s, a, r, s) we encounter.
  if we have reached terminal state, target = r
  if not, target = r + gamma*max(a)Q(s,a)
  target are scalars

## Linear Regression with one output
Targets and output are both scalars
W is a vector, b is a scalar
w = w - learning_rate*(gradient of w)
b = b - learning_rate*(gradient of b)
Keep repeting that untill our cost converges

## Linear Regression with multiple outputs - fits more to our case
  Conceptually, the target is only for prediction we actually made and 
  used Q(s,a), and not for an other actions. Therefore, any way its 
  corresponding to those actions should not be updated. Hence, the updates 
  should look like this, where we find the gradients only for the actions 
  we performed. And we update the weights corresponding that action:
      J = (r + gamma*max(a)Q(s,a) - Q(s,a))^2
      Wa = Wa - learning_rate*(gradient of Wa)
      ba = ba - learning_rate*(gradient of ba)

## I used a different implementation because it makes the code easier to extend 
(e.g. plug in Tensorflow). Example:
  Suppose K = 3 and we choose to perform a2 in the environment
  The error (and hence also the gradient) of W[a1] and W[a3] are 0
      _____________________________________
      |Prediction  |Target                |
      =====================================
      |Q(s,a1)     |Q(s,a1)               | -- so the error will be 0
      |Q(s,a2)     |r + gamma*max{Q(s,:)  | 
      |Q(s,a3)     |Q(s,a3)               | -- so the error will be 0
      _____________________________________
 
I will return only one chosen action to update. But in the with a Tensorflow 
or Scikit-learn targets should have the same shape as output prediction, 
therefore we must have K targets as well. We want an error for all other 
actions to be 0 and to achieve this we can simply equalize it with the 
prediction (as it shown in table above), so the weights for those actions 
wont be updated.

## Another small modification:
Instead of plain vanilla gradient descent, we will use gradient descent with 
momentum. Speeds up traning significantly. Basic idea - instead of taking a 
small step in the direction of the gradient on each iteration, we will keep 
around the old gradients in a term which we will call a velocity or the 
momentum. Its the number close to 1 (0.9, 0.99, etc.)

V(t) = momentum*V(t-1) - learning_rate*(gradient at time t)
W(t) = W(t-1) + V(t)

## Layout and Design
2 modes of operation: train and test 
train data must come before test data 
How this would work? 

## The main part of the code code will look something 
like this (Pseudocode):

env = Env()
agent = Agent()
portfolio_values = []
for _ in range(num_episodes):
  val = play_one_episode(agent, env)
  portfolio_values.append(val)
plot(portfolio_values)


## play_one_episode (Pseudocode):

def play_one_episode(agent, env):
  s = env.reset()
  done = False
  while not done:
      a = agent.get_action(s)
      s, r, done, info = env.step(a)
      if train_mode:
          agent.train(s, a, r, s, done)
      s = s
  return info[portfolio_val]


## Normalizing data 
Data is not yet normalized
Different parts of state have different ranges:
number of shares owned, stock prices, cash
Whenever we get a new state - we will have a scalar object (from sklearn),
which will take our state and standardize to have 0 mean, etc.
Pseudocode:

state = env.reset()
state = scaler.transform(state)
...
next_state, reward, done, info = env.step(action)
next_state = scaler.transform(next_state)


## Agent

class Agent:
  def __init__(self):
      self.model = Model()
  def get_action(self, s):
      # calculate Q(s,a), take the argmax over a
  def train(self, s, a, r, s, done):
      input = s
      target = r + gamma*max{Q(s,:)} or r if done = True
      self.model.sgd(input, target) # gradient descent w/ momentum
