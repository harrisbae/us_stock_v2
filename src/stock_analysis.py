def plot_signals_with_strength(df, signals, title, strength_plot=True):
    """신호와 강도를 함께 표시하는 차트를 그립니다."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1], gridspec_kw={'hspace': 0.3})
    
    # 메인 차트 (가격과 신호)
    ax1.plot(df.index, df['Close'], label='Close', color='blue', alpha=0.7)
    
    # 신호 마커 표시
    for signal in signals:
        if signal['type'] == 'buy':
            ax1.scatter(signal['date'], signal['price'], 
                       color='green', marker='^', s=100, label='Buy Signal')
        else:
            ax1.scatter(signal['date'], signal['price'], 
                       color='red', marker='v', s=100, label='Sell Signal')
    
    # 버티컬 라인과 날짜 텍스트
    for signal in signals:
        ax1.axvline(x=signal['date'], color='gray', linestyle='--', alpha=0.3)
        ax1.text(signal['date'], ax1.get_ylim()[0], 
                signal['date'].strftime('%Y-%m-%d'),
                rotation=45, ha='right', va='top')
    
    ax1.set_title(title)
    ax1.grid(True)
    
    # 범례 생성 (중복 제거)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    # 강도 서브플롯
    if strength_plot:
        strengths = [signal['strength'] for signal in signals]
        dates = [signal['date'] for signal in signals]
        
        ax2.bar(dates, strengths, color='purple', alpha=0.6)
        ax2.set_title('Signal Strength')
        ax2.grid(True)
        
        # 강도 값 표시
        for i, strength in enumerate(strengths):
            ax2.text(dates[i], strength, f'{strength:.2f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig 