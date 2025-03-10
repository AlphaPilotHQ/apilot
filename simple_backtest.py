import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# 读取数据
def load_data(file_path, days=7):
    print(f"正在读取数据: {file_path}")
    start_time = time.time()
    df = pd.read_csv(file_path)
    
    # 确保时间列格式正确
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    
    # 设置时间列为索引
    df.set_index('candle_begin_time', inplace=True)
    
    print(f"原始数据量: {len(df)} 条记录")
    
    # 只取最近days天的数据
    if days > 0:
        # 计算每天的数据量（假设是分钟级别数据）
        daily_bars = 24 * 60
        df = df.iloc[-days * daily_bars:]
        print(f"只取最近 {days} 天数据，剩余 {len(df)} 条记录")
    
    print(f"数据读取完成，处理后数据量: {len(df)} 条记录")
    print(f"数据读取耗时: {time.time() - start_time:.2f} 秒")
    return df

# 计算连续上涨和下跌的K线数量
def calculate_continuous_bars(df):
    print("开始计算连续上涨和下跌的K线数量...")
    start_time = time.time()
    
    # 计算价格变化
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # 初始化连续上涨和下跌的计数列
    df['up_streak'] = 0
    df['down_streak'] = 0
    
    # 使用向量化操作代替循环
    # 创建一个临时Series来跟踪连续上涨
    temp_up = pd.Series(0, index=df.index)
    mask_up = df['price_change'] > 0
    
    # 对于每个上涨的K线，计数加1；否则重置为0
    for i in range(1, len(df)):
        if mask_up.iloc[i]:
            temp_up.iloc[i] = temp_up.iloc[i-1] + 1
    
    df['up_streak'] = temp_up
    
    # 创建一个临时Series来跟踪连续下跌
    temp_down = pd.Series(0, index=df.index)
    mask_down = df['price_change'] < 0
    
    # 对于每个下跌的K线，计数加1；否则重置为0
    for i in range(1, len(df)):
        if mask_down.iloc[i]:
            temp_down.iloc[i] = temp_down.iloc[i-1] + 1
    
    df['down_streak'] = temp_down
    
    print(f"连续K线计算完成，耗时: {time.time() - start_time:.2f} 秒")
    print(f"最大连续上涨K线数: {df['up_streak'].max()}")
    print(f"最大连续下跌K线数: {df['down_streak'].max()}")
    return df

# 生成交易信号
def generate_signals(df, up_threshold=7, down_threshold=7):
    print(f"开始生成交易信号，上涨阈值: {up_threshold}，下跌阈值: {down_threshold}...")
    start_time = time.time()
    
    # 初始化信号列
    df['signal'] = 0
    
    # 生成买入信号（连续上涨达到阈值）
    df.loc[df['up_streak'] >= up_threshold, 'signal'] = 1
    
    # 生成卖出信号（连续下跌达到阈值）
    df.loc[df['down_streak'] >= down_threshold, 'signal'] = -1
    
    buy_signals = len(df[df['signal'] == 1])
    sell_signals = len(df[df['signal'] == -1])
    
    print(f"信号生成完成，耗时: {time.time() - start_time:.2f} 秒")
    print(f"买入信号数量: {buy_signals}")
    print(f"卖出信号数量: {sell_signals}")
    return df

# 回测策略
def backtest_strategy(df):
    print("开始回测策略...")
    start_time = time.time()
    
    # 初始化持仓和资金
    df['position'] = 0
    df['cash'] = 10000  # 初始资金
    df['holdings'] = 0  # 持仓价值
    df['total_assets'] = df['cash']  # 总资产
    
    position = 0
    trades = 0
    
    # 遍历数据进行回测
    for i in range(1, len(df)):
        # 默认继承前一天的持仓
        df.loc[df.index[i], 'position'] = position
        
        # 处理买入信号
        if df['signal'].iloc[i] == 1 and position == 0:
            # 全仓买入
            position = 1
            df.loc[df.index[i], 'position'] = position
            df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] - df['close'].iloc[i] * position
            trades += 1
            if trades <= 10:  # 只打印前10笔交易
                print(f"买入: {df.index[i]}, 价格: {df['close'].iloc[i]}")
        
        # 处理卖出信号
        elif df['signal'].iloc[i] == -1 and position == 1:
            # 全部卖出
            df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] + df['close'].iloc[i] * position
            position = 0
            df.loc[df.index[i], 'position'] = position
            trades += 1
            if trades <= 10:  # 只打印前10笔交易
                print(f"卖出: {df.index[i]}, 价格: {df['close'].iloc[i]}")
        
        else:
            # 无交易，继承前一天的现金
            df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1]
        
        # 计算持仓价值和总资产
        df.loc[df.index[i], 'holdings'] = df['close'].iloc[i] * df['position'].iloc[i]
        df.loc[df.index[i], 'total_assets'] = df['cash'].iloc[i] + df['holdings'].iloc[i]
        
        # 每处理10000条数据打印一次进度
        if i % 10000 == 0:
            print(f"已处理 {i}/{len(df)} 条数据，完成: {i/len(df)*100:.2f}%")
    
    print(f"回测完成，耗时: {time.time() - start_time:.2f} 秒")
    print(f"总交易次数: {trades}")
    return df

# 绘制回测结果
def plot_results(df):
    print("开始绘制回测结果...")
    start_time = time.time()
    
    plt.figure(figsize=(14, 8))
    
    # 绘制价格走势
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='价格')
    
    # 标记买入和卖出点
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', s=100, label='买入信号')
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', s=100, label='卖出信号')
    
    plt.title('SOL-USDT 价格走势与交易信号')
    plt.ylabel('价格')
    plt.legend()
    
    # 绘制资产曲线
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['total_assets'], label='策略资产')
    
    # 计算买入持有策略的资产曲线
    initial_investment = df['total_assets'].iloc[0]
    buy_hold_assets = initial_investment * (1 + df['buy_hold_returns']).cumprod()
    plt.plot(df.index, buy_hold_assets, label='买入持有策略')
    
    plt.title('策略资产曲线对比')
    plt.ylabel('资产')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('strategy_results.png')
    plt.close()
    
    print(f"绘图完成，耗时: {time.time() - start_time:.2f} 秒")
    print("回测结果图表已保存为 strategy_results.png")

# 计算策略绩效指标
def calculate_performance(df):
    print("开始计算策略绩效...")
    start_time = time.time()
    
    # 计算累计收益率
    total_return = (df['total_assets'].iloc[-1] / df['total_assets'].iloc[0]) - 1
    
    # 计算年化收益率（假设252个交易日）
    annual_return = (1 + total_return) ** (252 / len(df)) - 1
    
    # 计算最大回撤
    df['cummax'] = df['total_assets'].cummax()
    df['drawdown'] = (df['total_assets'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min()
    
    # 计算夏普比率（假设无风险利率为0）
    sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
    
    # 计算交易次数
    trades = len(df[df['signal'] != 0])
    
    # 输出结果
    print(f"绩效计算完成，耗时: {time.time() - start_time:.2f} 秒")
    print("\n===== 策略绩效 =====")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"交易次数: {trades}")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'trades': trades
    }

def main():
    total_start_time = time.time()
    print("===== 开始执行回测 =====")
    
    # 数据文件路径
    file_path = 'SOL-USDT.csv'
    
    # 加载数据，只使用最近7天的数据
    df = load_data(file_path, days=7)
    
    # 显示数据前几行
    print("\n数据预览:")
    print(df.head())
    
    # 计算连续上涨和下跌的K线数量
    df = calculate_continuous_bars(df)
    
    # 生成交易信号
    df = generate_signals(df)
    
    # 回测策略
    df = backtest_strategy(df)
    
    # 计算策略收益率
    df['strategy_returns'] = df['total_assets'].pct_change()
    
    # 计算买入持有收益率
    df['buy_hold_returns'] = df['close'].pct_change()
    
    # 计算策略绩效
    performance = calculate_performance(df)
    
    # 绘制回测结果
    plot_results(df)
    
    # 保存结果到CSV
    df.to_csv('backtest_results.csv')
    print("回测结果已保存到 backtest_results.csv")
    
    print(f"\n===== 回测执行完毕，总耗时: {time.time() - total_start_time:.2f} 秒 =====")

if __name__ == "__main__":
    main() 