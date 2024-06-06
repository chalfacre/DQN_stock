import numpy as np
import pandas as pd
import yfinance as yf
import ta
import gym
from gym import spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class StockTradingEnv(gym.Env):
    def __init__(self, stock_symbol, start_date='2001-01-01', end_date='2024-05-24'):
        super(StockTradingEnv, self).__init__()
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df = self.load_data(stock_symbol, start_date, end_date)
        self.n_days = len(self.df)
        self.current_day = 0
        self.balance = 10000
        self.shares_held = 0
        self.history = []

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(24,), dtype=np.float32)

    def load_data(self, stock_symbol, start_date, end_date):
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        df['10d_MA'] = df['Close'].rolling(window=10).mean()
        df['50d_MA'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()

        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()

        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch'] = stoch.stoch()

        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['VROC'] = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()
        df['PROC'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc()

        df.dropna(inplace=True)
        return df

    def reset(self):
        self.current_day = 0
        self.balance = 10000
        self.shares_held = 0
        self.history = [self.df['Close'].iloc[i] for i in range(10)]
        self.update_state()
        return self.state

    def update_state(self):
        self.state = np.concatenate([
            np.array(self.history[-10:]),  # Last 10 days of closing prices
            np.array([
                self.balance,
                self.shares_held,
                self.df['10d_MA'].iloc[self.current_day],
                self.df['50d_MA'].iloc[self.current_day],
                self.df['RSI'].iloc[self.current_day],
                self.df['MACD'].iloc[self.current_day],
                self.df['MACD_Signal'].iloc[self.current_day],
                self.df['Bollinger_High'].iloc[self.current_day],
                self.df['Bollinger_Low'].iloc[self.current_day],
                self.df['Stoch'].iloc[self.current_day],
                self.df['ATR'].iloc[self.current_day],
                self.df['OBV'].iloc[self.current_day],
                self.df['VROC'].iloc[self.current_day],
                self.df['PROC'].iloc[self.current_day]
            ])
        ]).flatten()

    def step(self, action):
        self.current_day += 1
        if self.current_day >= self.n_days:
            self.current_day = 0

        self.history.append(self.df['Close'].iloc[self.current_day])
        if len(self.history) > 10:
            self.history.pop(0)

        self.update_state()
        reward = 0

        if action == 0:  # Sell
            reward = self.shares_held * self.df['Close'].iloc[self.current_day]
            self.balance += reward
            self.shares_held = 0

        elif action == 2:  # Buy
            shares_to_buy = self.balance // self.df['Close'].iloc[self.current_day]
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * self.df['Close'].iloc[self.current_day]

        done = self.current_day == self.n_days - 1

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Day: {self.current_day}, Price: {self.df['Close'].iloc[self.current_day]}, Balance: {self.balance}, Shares Held: {self.shares_held}")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def train_agent(stock_symbol, episodes=1000, batch_size=32):
    env = StockTradingEnv(stock_symbol)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # Save the model weights at the end of each episode
        agent.save(f"dqn_model_{stock_symbol}.h5")

def predict_next_10_days(stock_symbol):
    env = StockTradingEnv(stock_symbol)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Load the trained model
    agent.load(f"dqn_model_{stock_symbol}.h5")

    state = env.reset()

# Example usage:
# Train the agent for a specific stock
train_agent(stock_symbol='AAPL', episodes=1000, batch_size=32)

# Predict the next 10 days of stock prices for the same stock
predicted_prices = predict_next_10_days(stock_symbol='AAPL')
print("Predicted prices for the next 10 days for AAPL:", predicted_prices)

