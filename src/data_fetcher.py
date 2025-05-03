import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import numpy as np

# pykrx 라이브러리 (한국 주식 데이터 위한 라이브러리)
try:
    from pykrx import stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    print("WARNING: pykrx 라이브러리가 설치되지 않았습니다. 한국 주식의 외국인 거래량 정보를 가져올 수 없습니다.")

class DataFetcher:
    def __init__(self):
        self.data = None
        self.ticker = None
        self.is_korean_stock = False
        
    def fetch_data(self, ticker, period="1y", interval="1d", start_date=None, end_date=None):
        """
        Yahoo Finance에서 주가 데이터를 가져옵니다.
        
        Args:
            ticker (str): 종목 코드 (예: AAPL, 005930.KS)
            period (str): 데이터 기간 (1d, 1w, 1m, 1y)
            interval (str): 데이터 간격 (1m, 1h, 1d)
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        self.ticker = ticker
        
        # 한국 주식인지 확인 (.KS, .KQ로 끝나거나 숫자로만 구성된 코드)
        self.is_korean_stock = ticker.endswith(('.KS', '.KQ')) or ticker.isdigit()
        
        try:
            stock_yf = yf.Ticker(ticker)
            
            # start_date와 end_date가 모두 제공된 경우, period 대신 날짜 범위 사용
            if start_date and end_date:
                self.data = stock_yf.history(start=start_date, end=end_date, interval=interval)
                print(f"날짜 범위 사용: {start_date} ~ {end_date}")
            else:
                self.data = stock_yf.history(period=period, interval=interval)
            
            # 데이터 검증
            if self.data.empty:
                print(f"WARNING: {ticker}에 대한 데이터가 없습니다.")
                return None
            
            # 인덱스가 datetime인지 확인
            if not isinstance(self.data.index, pd.DatetimeIndex):
                self.data.index = pd.to_datetime(self.data.index)
                
            # 컬럼명 재설정
            self.data.columns = [col.lower().capitalize() for col in self.data.columns]
            
            # 한국 주식이고 일별 데이터인 경우 외국인 거래량 추가
            if self.is_korean_stock and interval == '1d' and PYKRX_AVAILABLE:
                self._add_foreign_investor_data(start_date, end_date)
            
            return self.data
            
        except Exception as e:
            print(f"ERROR: {ticker} 데이터 수집 중 오류 발생: {e}")
            return None
    
    def _add_foreign_investor_data(self, start_date=None, end_date=None):
        """
        한국 주식의 외국인 거래량 데이터를 추가합니다.
        
        Args:
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
        """
        if not PYKRX_AVAILABLE:
            print("WARNING: pykrx 라이브러리가 설치되지 않아 외국인 거래량 정보를 가져올 수 없습니다.")
            return
            
        try:
            # 한국 종목코드 추출 (XXXXX.KS, XXXXX.KQ 형식에서 XXXXX 부분만)
            if '.' in self.ticker:
                krx_ticker = self.ticker.split('.')[0]
            else:
                krx_ticker = self.ticker
                
            # 6자리가 아닌 경우 앞에 0 추가
            krx_ticker = krx_ticker.zfill(6)
            
            # 기간 설정
            if start_date is None:
                start_date = self.data.index[0].strftime('%Y%m%d')
            else:
                start_date = start_date.replace('-', '')
                
            if end_date is None:
                end_date = self.data.index[-1].strftime('%Y%m%d')
            else:
                end_date = end_date.replace('-', '')
                
            print(f"외국인 거래량 데이터 가져오는 중: {krx_ticker} ({start_date}~{end_date})")
            
            try:
                # 최신 pykrx API에 맞춰서 외국인 순매수 데이터 가져오기
                foreign_data = stock.get_market_net_purchases_of_equities_by_ticker(
                    start_date, 
                    end_date,
                    investor_code='FORN'  # 외국인 투자자 코드
                )
                
                # 종목 코드로 필터링
                if krx_ticker in foreign_data.index:
                    # 해당 종목의 외국인 순매수 데이터만 추출
                    foreign_buy_series = foreign_data.loc[krx_ticker]
                    foreign_buy = pd.DataFrame(foreign_buy_series).reset_index()
                    foreign_buy.columns = ['Date', 'Foreign_Buy']
                    
                    # 날짜 형식 변환
                    foreign_buy['Date'] = pd.to_datetime(foreign_buy['Date'])
                    foreign_buy = foreign_buy.set_index('Date')
                    
                    # 기존 데이터와 합치기 (날짜 기준 조인)
                    self.data = self.data.merge(
                        foreign_buy, 
                        left_index=True, 
                        right_index=True, 
                        how='left'
                    )
                    
                    # NaN 값 0으로 대체
                    self.data['Foreign_Buy'] = self.data['Foreign_Buy'].fillna(0)
                    
                    # 외국인 순매수 비율 계산 (거래량 대비)
                    self.data['Foreign_Buy_Ratio'] = self.data['Foreign_Buy'] / (self.data['Volume'] * self.data['Close'])
                    
                    # 외국인 매수 신호 계산
                    self.data['Foreign_Buy_Signal'] = np.where(
                        self.data['Foreign_Buy'] > 0,  # 외국인 순매수
                        1, 0
                    )
                    
                    print(f"외국인 거래량 데이터 추가 완료")
                else:
                    print(f"WARNING: 종목코드 {krx_ticker}에 대한 외국인 거래량 데이터를 찾을 수 없습니다.")
            
            except:
                # 대체 방법: 일별 순매수 정보 사용
                foreign_data = pd.DataFrame()
                
                # 일별로 외국인 순매수 데이터 수집
                for date in pd.date_range(start=pd.to_datetime(start_date, format='%Y%m%d'), 
                                          end=pd.to_datetime(end_date, format='%Y%m%d')):
                    date_str = date.strftime('%Y%m%d')
                    try:
                        daily_data = stock.get_market_trading_value_by_investor(date_str, investor='외국인')
                        if krx_ticker in daily_data.index:
                            daily_value = daily_data.loc[krx_ticker, '순매수']
                            foreign_data.loc[date, 'Foreign_Buy'] = daily_value
                    except:
                        continue
                
                if not foreign_data.empty:
                    # 기존 데이터와 합치기
                    self.data = self.data.merge(
                        foreign_data, 
                        left_index=True, 
                        right_index=True, 
                        how='left'
                    )
                    
                    # NaN 값 0으로 대체
                    self.data['Foreign_Buy'] = self.data['Foreign_Buy'].fillna(0)
                    
                    # 외국인 순매수 비율 계산 (거래량 대비)
                    self.data['Foreign_Buy_Ratio'] = self.data['Foreign_Buy'] / (self.data['Volume'] * self.data['Close'])
                    
                    # 외국인 매수 신호 계산
                    self.data['Foreign_Buy_Signal'] = np.where(
                        self.data['Foreign_Buy'] > 0,  # 외국인 순매수
                        1, 0
                    )
                    
                    print(f"외국인 거래량 데이터 추가 완료 (대체 방법)")
                else:
                    print(f"WARNING: 외국인 거래량 데이터를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"ERROR: 외국인 거래량 데이터 수집 중 오류 발생: {e}")
            # 실패해도 기본 데이터는 추가
            self.data['Foreign_Buy'] = 0
            self.data['Foreign_Buy_Ratio'] = 0
            self.data['Foreign_Buy_Signal'] = 0
            print("외국인 거래량 데이터 없이 계속 진행합니다.")
    
    def save_data(self, output_dir=None):
        """
        수집된 데이터를 CSV 파일로 저장합니다.
        
        Args:
            output_dir (str): 저장할 디렉토리 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.data is None or self.ticker is None:
            print("ERROR: 저장할 데이터가 없습니다.")
            return None
            
        # 기본 출력 디렉토리 설정
        if output_dir is None:
            today = datetime.now().strftime("%Y%m%d")
            output_dir = os.path.join("output", today, self.ticker)
            
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일 저장
        output_file = os.path.join(output_dir, f"{self.ticker}_data.csv")
        self.data.to_csv(output_file)
        print(f"데이터가 {output_file}에 저장되었습니다.")
        
        return output_file

if __name__ == "__main__":
    # 테스트 코드
    fetcher = DataFetcher()
    data = fetcher.fetch_data("AAPL", period="1m", interval="1d")
    if data is not None:
        print(data.head())
        fetcher.save_data() 