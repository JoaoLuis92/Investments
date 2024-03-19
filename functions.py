# FINANCIAL INDICATORS
# I need to separate this between functions to compute the indicators and functions to check the signals.
# for example, check trend signal, check oscillator signal, etc, which make use of the financial indicators!


# Simple moving average

def simple_moving_average(df, period):

    df["SMA"+str(period)] = df["Close"].rolling(period).mean()


# Exponential moving average

def exponential_moving_average(df, period):

    df["EMA"+str(period)] = df['Close'].ewm(span = period).mean()

# Full Stochastic Oscillator

def full_stochastic(df, fk_period, sk_period, sd_period):

    fast_k_list = []

    for i in range(len(df)):
        low = df.iloc[i]['Low']
        high = df.iloc[i]['High']

        if i >= fk_period:

            for n in range(fk_period):

                if df.iloc[i-n]['High'] >= high:
                    high = df.iloc[i-n]['High']
                elif df.iloc[i-n]['Low'] < low:
                    low = df.iloc[i-n]['Low']
        if high != low:
            fast_k = 100 * (df.iloc[i]['Close'] - low) / (high - low)
        else:
            fast_k = 0

        fast_k_list.append(fast_k)

    df["Fast K"] = fast_k_list
    df["Slow K"] = df["Fast K"].rolling(sk_period).mean()
    df["Slow D"] = df["Slow K"].rolling(sd_period).mean()
