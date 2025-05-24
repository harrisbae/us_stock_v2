def plot_stock_data(df, symbol, period, interval, output_dir):
    """주식 데이터를 시각화하는 함수"""
    try:
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'AppleGothic'  # macOS
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif platform.system() == 'Linux':
            plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 그래프 스타일 설정
        plt.style.use('dark_background')
        
        # 그래프 크기 설정
        fig = plt.figure(figsize=(20, 10))
        
        # 메인 차트 (가격과 HMA)
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        
        # 가격 데이터 플롯
        ax1.plot(df.index, df['Close'], label='Close', color='white', alpha=0.3)
        
        # HMA 플롯
        ax1.plot(df.index, df['HMA'], label='HMA', color='yellow', linewidth=2)
        
        # 만트라 밴드 플롯
        ax1.plot(df.index, df['Upper Band'], label='Upper Band', color='red', alpha=0.5)
        ax1.plot(df.index, df['Lower Band'], label='Lower Band', color='green', alpha=0.5)
        
        # 볼린저 밴드 플롯
        ax1.plot(df.index, df['BB_Upper'], label='BB Upper', color='red', linestyle='--', alpha=0.3)
        ax1.plot(df.index, df['BB_Lower'], label='BB Lower', color='green', linestyle='--', alpha=0.3)
        
        # 이동평균선 플롯
        ax1.plot(df.index, df['MA20'], label='MA20', color='blue', alpha=0.5)
        ax1.plot(df.index, df['MA60'], label='MA60', color='purple', alpha=0.5)
        ax1.plot(df.index, df['MA120'], label='MA120', color='orange', alpha=0.5)
        ax1.plot(df.index, df['MA200'], label='MA200', color='cyan', alpha=0.5)
        
        # 신호 마커 추가
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        # 매수 신호
        ax1.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', color='red', s=100, label='Buy Signal')
        
        # 매도 신호
        ax1.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', color='green', s=100, label='Sell Signal')
        
        # 신호 발생일 표시
        for idx in buy_signals.index:
            ax1.text(idx, buy_signals.loc[idx, 'Close'], 
                    f'매수\n{buy_signals.loc[idx, "Close"]:.1f}',
                    color='red', ha='center', va='bottom')
            
        for idx in sell_signals.index:
            ax1.text(idx, sell_signals.loc[idx, 'Close'], 
                    f'매도\n{sell_signals.loc[idx, "Close"]:.1f}',
                    color='green', ha='center', va='top')
        
        # 신호 강도 표시
        for idx in df[df['Signal'] != 0].index:
            strength = df.loc[idx, 'Signal_Strength']
            if strength > 0:
                ax1.text(idx, df.loc[idx, 'Close'], 
                        f'강도: {strength:.1f}',
                        color='red', ha='right', va='bottom')
            else:
                ax1.text(idx, df.loc[idx, 'Close'], 
                        f'강도: {abs(strength):.1f}',
                        color='green', ha='right', va='top')
        
        # 신호 발생일마다 버티컬 라인 추가
        for idx in df[df['Signal'] != 0].index:
            ax1.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
        
        # y축 범위 설정
        price_range = df['Close'].max() - df['Close'].min()
        ax1.set_ylim(df['Close'].min() - price_range * 0.1, 
                     df['Close'].max() + price_range * 0.1)
        
        # 그리드 설정
        ax1.grid(True, alpha=0.3)
        
        # 레이블 설정
        ax1.set_title(f'{symbol} Stock Price Analysis ({period}, {interval})', fontsize=15)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        
        # x축 날짜 포맷 설정
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # 범례 설정
        ax1.legend(loc='upper left')
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 그래프 저장
        output_file = os.path.join(output_dir, f'{symbol}_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"그래프가 저장되었습니다: {output_file}")
        
    except Exception as e:
        print(f"그래프 생성 중 오류 발생: {str(e)}")
        print(f"오류 타입: {type(e).__name__}")
        import traceback
        print(f"스택 트레이스: {traceback.format_exc()}")
        plt.close() 