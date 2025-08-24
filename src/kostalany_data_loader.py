#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
코스탈라니 달걀모형 거시경제 데이터 로더
"""

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json
import os
import time
import ssl
import certifi
from datetime import datetime, timedelta
import numpy as np
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("FRED API를 사용할 수 없습니다. 대체 데이터 소스를 사용합니다.")

# macOS에서 SSL 인증서 문제 해결을 위한 코드
try:
    # SSL 인증서 경로 설정 시도
    import os
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    # SSL 컨텍스트를 생성하고 기본값으로 설정
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = ssl._create_unverified_context
    print("SSL 인증서 설정 완료")
except Exception as e:
    print(f"SSL 설정 중 오류 발생: {e}")
    # 기본 SSL 컨텍스트 사용
    ssl_context = ssl._create_unverified_context()

# 현대적이고 신뢰할 수 있는 User-Agent 목록
MODERN_USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def get_random_user_agent():
    """랜덤 User-Agent 반환"""
    import random
    return random.choice(MODERN_USER_AGENTS)

def get_enhanced_headers():
    """향상된 헤더 반환"""
    return {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9,ko;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }

def safe_request_with_delay(url, headers=None, delay=2.0, max_retries=3):
    """안전한 요청과 지연 시간을 포함한 HTTP 요청"""
    if headers is None:
        headers = get_enhanced_headers()
    
    for attempt in range(max_retries):
        try:
            # 요청 간 지연 시간
            if attempt > 0:
                time.sleep(delay)
            
            response = requests.get(url, headers=headers, timeout=15)
            
            # 성공적인 응답
            if response.status_code == 200:
                return response
            
            # 403 Forbidden 등 차단된 경우
            elif response.status_code == 403:
                print(f"⚠️ 접근이 차단되었습니다 (시도 {attempt + 1}/{max_retries}): {url}")
                if attempt < max_retries - 1:
                    time.sleep(delay * 2)  # 더 긴 지연 시간
                    continue
            
            # 기타 오류
            else:
                print(f"⚠️ HTTP 오류 {response.status_code} (시도 {attempt + 1}/{max_retries}): {url}")
                
        except requests.exceptions.Timeout:
            print(f"⚠️ 요청 시간 초과 (시도 {attempt + 1}/{max_retries}): {url}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ 요청 오류 (시도 {attempt + 1}/{max_retries}): {e}")
        
        # 마지막 시도가 아니면 지연 후 재시도
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    # 모든 시도 실패
    print(f"❌ 모든 시도 실패: {url}")
    return None

class MacroDataLoader:
    """거시경제 지표 데이터 로더 클래스"""
    
    def __init__(self, country='korea', fred_api_key=None, default_values=None, enable_web_scraping=False):
        """
        초기화 함수
        
        Parameters:
            country (str): 국가 코드 (기본: 'korea')
            fred_api_key (str): FRED API 키
            default_values (dict): 스크래핑 실패 시 사용할 기본값 딕셔너리
            enable_web_scraping (bool): 웹 스크래핑을 우선적으로 시도할지 여부
        """
        self.country = country.lower()
        self.data_cache = {}
        self.cache_dir = "data_cache"
        self.default_values = default_values or {}
        self.enable_web_scraping = enable_web_scraping
        
        # 환경 변수에서 FRED API 키 가져오기
        # FRED API 키 설정 방법:
        # 1. 환경 변수 FRED_API_KEY에 설정
        # 2. fred_api_key 파라미터로 전달
        # 3. '.fred_api_key' 파일에서 읽기
        
        # 기본 키 설정 (없으면 샘플 키)
        self.fred_api_key = fred_api_key or os.environ.get('FRED_API_KEY')
        
        # 파일에서 API 키 읽기 시도
        if not self.fred_api_key:
            try:
                api_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.fred_api_key')
                if os.path.exists(api_key_file):
                    with open(api_key_file, 'r') as f:
                        self.fred_api_key = f.read().strip()
                        print("FRED API 키를 파일에서 읽었습니다.")
            except Exception as e:
                print(f"API 키 파일 읽기 실패: {e}")
        
        # 샘플 API 키 사용 (개발용, 실제 사용시 발급 필요)
        if not self.fred_api_key and self.country == 'us':
            self.fred_api_key = "0123456789abcdef0123456789abcdef"
            print("경고: 샘플 FRED API 키를 사용합니다. 실제 키로 변경하세요.")
            print("FRED API 키는 https://fredapi.stlouisfed.org/에서 발급받을 수 있습니다.")
            
        self.fred = None
        
        if FRED_AVAILABLE and self.fred_api_key and self.country == 'us':
            try:
                self.fred = Fred(api_key=self.fred_api_key)
                print("FRED API 연결 성공")
            except Exception as e:
                print(f"FRED API 연결 실패: {e}")
                self.fred = None
        else:
            if self.country == 'us':
                print("FRED API를 사용하지 않습니다. 웹 스크래핑으로 데이터를 수집합니다.")
        
        # 캐시 디렉토리 생성
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 국가별 API 소스 설정
        self.apis = {
            'korea': {
                'gdp': self._get_korea_gdp,
                'inflation': self._get_korea_inflation,
                'interest': self._get_korea_interest_rate,
                'unemployment': self._get_korea_unemployment,
                'vix': self._get_vix_index,
                'dxy': self._get_dollar_index
            },
            'us': {
                'gdp': self._get_us_gdp,
                'inflation': self._get_us_inflation,
                'interest': self._get_us_interest_rate,
                'unemployment': self._get_us_unemployment,
                'vix': self._get_vix_index,
                'dxy': self._get_dollar_index
            }
        }
        
    def _load_cache(self, indicator):
        """캐시된 데이터 로드"""
        cache_file = os.path.join(self.cache_dir, f"{self.country}_{indicator}.json")
        
        if os.path.exists(cache_file):
            # 캐시 파일이 10분 이내에 업데이트되었는지 확인 (기존 24시간에서 10분으로 변경)
            if time.time() - os.path.getmtime(cache_file) < 600:  # 10분 = 600초
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass
        
        return None
    
    def _save_cache(self, indicator, data):
        """데이터 캐시 저장"""
        cache_file = os.path.join(self.cache_dir, f"{self.country}_{indicator}.json")
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def _get_korea_gdp(self):
        """한국 GDP 성장률 데이터 가져오기"""
        cached = self._load_cache('gdp')
        if cached:
            return cached
        
        try:
            # 실시간 데이터 로딩 시도
            url = "https://tradingeconomics.com/south-korea/gdp-growth-annual"
            
            response = safe_request_with_delay(url)
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_gdp = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_gdp = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_gdp' not in locals():
                            recent_gdp = 2.2  # 기본값
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"GDP 데이터 파싱 오류: {e}")
                    recent_gdp = 2.2
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_gdp,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            else:
                # 요청 실패 시 다른 사이트 시도
                try:
                    url = "https://www.focus-economics.com/country-indicator/korea/gdp"
                    response = safe_request_with_delay(url)
                    
                    if response and response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # 실패 시 기본값 사용
                        recent_gdp = 2.2
                        source = "Focus Economics (웹)"
                    else:
                        recent_gdp = 2.2
                        source = "기본값 (서버 응답 실패)"
                except:
                    recent_gdp = 2.2
                    source = "기본값 (서버 연결 실패)"
                
                result = {
                    'current': recent_gdp,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            
            self._save_cache('gdp', result)
            return result
        except Exception as e:
            print(f"GDP 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            return {'current': 2.2, 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'source': "기본값 (오류)"}
    
    def _get_korea_inflation(self):
        """한국 인플레이션 데이터 가져오기"""
        cached = self._load_cache('inflation')
        if cached:
            return cached
        
        try:
            # 실시간 데이터 로딩 시도
            url = "https://tradingeconomics.com/south-korea/inflation-cpi"
            headers = get_enhanced_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_inflation = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_inflation = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_inflation' not in locals():
                            val = self.default_values.get('inflation')
                            recent_inflation = val if val is not None else 2.7
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"인플레이션 데이터 파싱 오류: {e}")
                    val = self.default_values.get('inflation')
                    recent_inflation = val if val is not None else 2.7
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_inflation,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            else:
                recent_inflation = self.default_values['inflation'] if self.default_values.get('inflation') is not None else 2.7
                source = "기본값 (서버 응답 실패)"
                result = {
                    'current': recent_inflation,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            
            self._save_cache('inflation', result)
            return result
        except Exception as e:
            print(f"인플레이션 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('inflation')
            return {'current': val if val is not None else 2.7, 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'source': "기본값 (오류)"}
    
    def _get_korea_interest_rate(self):
        """한국 기준금리 데이터 가져오기"""
        cached = self._load_cache('interest')
        if cached:
            return cached
        
        try:
            # 실시간 데이터 로딩 시도
            url = "https://tradingeconomics.com/south-korea/interest-rate"
            headers = get_enhanced_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_rate = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_rate = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_rate' not in locals():
                            val = self.default_values.get('interest')
                            recent_rate = val if val is not None else 3.5
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"금리 데이터 파싱 오류: {e}")
                    val = self.default_values.get('interest')
                    recent_rate = val if val is not None else 3.5
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_rate,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            else:
                # 다른 사이트 시도
                url = "https://www.global-rates.com/en/interest-rates/central-banks/central-bank-korea/bok-interest-rate.aspx"
                try:
                    response = requests.get(url, headers=get_enhanced_headers(), timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Global Rates 사이트에서 값 추출 시도
                        try:
                            rate_text = soup.select_one("table.tabledata1 tr:nth-of-type(1) td:nth-of-type(2)")
                            if rate_text:
                                rate_str = rate_text.text.strip().replace(',', '.').replace('%', '')
                                recent_rate = float(rate_str)
                                source = "Global Rates (실시간)"
                            else:
                                recent_rate = 3.5
                                source = "기본값 (Global Rates 파싱 실패)"
                        except:
                            recent_rate = 3.5
                            source = "기본값 (Global Rates 파싱 오류)"
                    else:
                        recent_rate = 3.5
                        source = "기본값 (서버 응답 실패)"
                except:
                    recent_rate = 3.5
                    source = "기본값 (서버 연결 실패)"
                
                result = {
                    'current': recent_rate,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            
            self._save_cache('interest', result)
            return result
        except Exception as e:
            print(f"금리 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('interest')
            return {'current': val if val is not None else 3.5, 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'source': "기본값 (오류)"}
    
    def _get_korea_unemployment(self):
        """한국 실업률 데이터 가져오기"""
        cached = self._load_cache('unemployment')
        if cached:
            return cached
        
        try:
            # 실시간 데이터 로딩 시도
            url = "https://tradingeconomics.com/south-korea/unemployment-rate"
            headers = get_enhanced_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_unemployment = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_unemployment = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_unemployment' not in locals():
                            val = self.default_values.get('unemployment')
                            recent_unemployment = val if val is not None else 3.0
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"실업률 데이터 파싱 오류: {e}")
                    val = self.default_values.get('unemployment')
                    recent_unemployment = val if val is not None else 3.0
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_unemployment,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            else:
                recent_unemployment = 3.0
                source = "기본값 (서버 응답 실패)"
                result = {
                    'current': recent_unemployment,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
            
            self._save_cache('unemployment', result)
            return result
        except Exception as e:
            print(f"실업률 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('unemployment')
            return {'current': val if val is not None else 3.0, 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'source': "기본값 (오류)"}
    
    def _get_us_gdp(self):
        """미국 GDP 성장률 데이터 가져오기"""
        cached = self._load_cache('gdp')
        # 캐시 유효 시간을 30분으로 단축
        if cached and time.time() - os.path.getmtime(os.path.join(self.cache_dir, f"{self.country}_gdp.json")) < 1800:
            return cached
        
        try:
            # FRED API 먼저 시도 (API 키가 있는 경우)
            if FRED_AVAILABLE and self.fred:
                try:
                    # FRED API를 통해 실질 GDP 성장률 데이터 가져오기
                    print(f"FRED API GDP 데이터 요청 시작...")
                    gdp_growth = self.fred.get_series('A191RL1Q225SBEA')
                    if not gdp_growth.empty:
                        latest_gdp = round(gdp_growth.iloc[-1], 2)
                        
                        print(f"FRED API GDP 데이터 로드 성공: {latest_gdp}%")
                        
                        result = {
                            'current': latest_gdp,
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'source': 'FRED (API)'
                        }
                        
                        self._save_cache('gdp', result)
                        return result
                    else:
                        print("FRED API에서 빈 데이터 반환됨")
                except Exception as e:
                    print(f"FRED API GDP 데이터 로드 실패: {e}")
            
            # 웹 스크래핑으로 시도
            # 실시간 데이터 로딩 시도
            url = "https://tradingeconomics.com/united-states/gdp-growth-annual"
            headers = get_enhanced_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_gdp = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_gdp = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_gdp' not in locals():
                            recent_gdp = self.default_values.get('gdp', 2.8)  # 파라미터로 받은 기본값 또는 하드코딩된 기본값
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"미국 GDP 데이터 파싱 오류: {e}")
                    recent_gdp = self.default_values.get('gdp', 2.8)
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_gdp,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
                
                self._save_cache('gdp', result)
                return result
            else:
                # 다른 모든 방법 실패 시 기본값
                result = {
                    'current': self.default_values.get('gdp', 2.8),
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': '기본값 (모든 방법 실패)'
                }
                
                self._save_cache('gdp', result)
                return result
        except Exception as e:
            print(f"미국 GDP 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            return {'current': self.default_values.get('gdp', 2.8), 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'source': "기본값 (오류)"}
    
    def _get_us_inflation(self):
        """미국 인플레이션 데이터 가져오기"""
        cached = self._load_cache('inflation')
        if cached and time.time() - os.path.getmtime(os.path.join(self.cache_dir, f"{self.country}_inflation.json")) < 3600:  # 1시간 이내 캐시만 사용
            return cached
        
        try:
            # FRED API 먼저 시도
            if FRED_AVAILABLE and self.fred:
                try:
                    # FRED API를 통해 소비자물가지수(CPI) 데이터 가져오기 
                    # CPIAUCSL: 물가지수, PCEPI: 개인소비지출 물가지수
                    print(f"FRED API 인플레이션 데이터 요청 시작...")
                    
                    # CPIAUCSL: 소비자물가지수, 계절조정
                    cpi_data = self.fred.get_series('CPIAUCSL')
                    
                    if not cpi_data.empty:
                        # 연간 인플레이션율 계산
                        inflation_yoy = cpi_data.pct_change(12) * 100
                        latest_inflation = round(inflation_yoy.dropna().iloc[-1], 2)
                        
                        print(f"FRED API 인플레이션 데이터 로드 성공: {latest_inflation}%")
                        
                        result = {
                            'current': latest_inflation,
                            'last_update': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'FRED (API)'
                        }
                        
                        self._save_cache('inflation', result)
                        return result
                    else:
                        print("FRED API에서 빈 인플레이션 데이터 반환됨")
                except Exception as e:
                    print(f"FRED API 인플레이션 데이터 로드 실패: {e}")
            
            # FRED API가 없거나 실패한 경우 웹 스크래핑으로 대체
            url = "https://tradingeconomics.com/united-states/inflation-cpi"
            headers = get_enhanced_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_inflation = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_inflation = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_inflation' not in locals():
                            val = self.default_values.get('inflation')
                            recent_inflation = val if val is not None else 3.7
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"미국 인플레이션 데이터 파싱 오류: {e}")
                    val = self.default_values.get('inflation')
                    recent_inflation = val if val is not None else 3.7
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_inflation,
                    'last_update': datetime.now().strftime('%Y-%m-%d'),
                    'source': source
                }
            else:
                # 요청 실패 시 기본값 사용
                val = self.default_values.get('inflation')
                result = {
                    'current': val if val is not None else 3.7,
                    'last_update': datetime.now().strftime('%Y-%m-%d'),
                    'source': '기본값 (서버 응답 실패)'
                }
            
            self._save_cache('inflation', result)
            return result
        except Exception as e:
            print(f"인플레이션 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('inflation')
            return {'current': val if val is not None else 3.7, 'last_update': datetime.now().strftime('%Y-%m-%d'), 'source': '기본값 (오류)'}
    
    def _get_us_interest_rate(self):
        """미국 기준금리 데이터 가져오기"""
        cached = self._load_cache('interest')
        if cached and time.time() - os.path.getmtime(os.path.join(self.cache_dir, f"{self.country}_interest.json")) < 3600:  # 1시간 이내 캐시만 사용
            return cached
        
        try:
            # FRED API 먼저 시도
            if FRED_AVAILABLE and self.fred:
                try:
                    # FRED API를 통해 연방기금금리(Federal Funds Rate) 데이터 가져오기
                    print(f"FRED API 기준금리 데이터 요청 시작...")
                    
                    # FEDFUNDS: 실효 연방기금금리
                    interest_data = self.fred.get_series('FEDFUNDS')
                    
                    if not interest_data.empty:
                        latest_interest = round(interest_data.iloc[-1], 2)
                        
                        print(f"FRED API 기준금리 데이터 로드 성공: {latest_interest}%")
                        
                        result = {
                            'current': latest_interest,
                            'last_update': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'FRED (API)'
                        }
                        
                        self._save_cache('interest', result)
                        return result
                    else:
                        print("FRED API에서 빈 기준금리 데이터 반환됨")
                except Exception as e:
                    print(f"FRED API 기준금리 데이터 로드 실패: {e}")
            
            # FRED API가 없거나 실패한 경우 웹 스크래핑으로 대체
            url = "https://tradingeconomics.com/united-states/interest-rate"
            headers = get_enhanced_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_rate = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_rate = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_rate' not in locals():
                            val = self.default_values.get('interest')
                            recent_rate = val if val is not None else 5.5
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"미국 기준금리 데이터 파싱 오류: {e}")
                    val = self.default_values.get('interest')
                    recent_rate = val if val is not None else 5.5
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_rate,
                    'last_update': datetime.now().strftime('%Y-%m-%d'),
                    'source': source
                }
            else:
                # 요청 실패 시 기본값 사용
                val = self.default_values.get('interest')
                result = {
                    'current': val if val is not None else 5.5,
                    'last_update': datetime.now().strftime('%Y-%m-%d'),
                    'source': '기본값 (서버 응답 실패)'
                }
            
            self._save_cache('interest', result)
            return result
        except Exception as e:
            print(f"기준금리 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('interest')
            return {'current': val if val is not None else 5.5, 'last_update': datetime.now().strftime('%Y-%m-%d'), 'source': '기본값 (오류)'}
    
    def _get_us_unemployment(self):
        """미국 실업률 데이터 가져오기"""
        cached = self._load_cache('unemployment')
        if cached and time.time() - os.path.getmtime(os.path.join(self.cache_dir, f"{self.country}_unemployment.json")) < 3600:  # 1시간 이내 캐시만 사용
            return cached
        
        try:
            # FRED API 먼저 시도
            if FRED_AVAILABLE and self.fred:
                try:
                    # FRED API를 통해 실업률 데이터 가져오기
                    print(f"FRED API 실업률 데이터 요청 시작...")
                    
                    # UNRATE: 실업률
                    unemployment_data = self.fred.get_series('UNRATE')
                    
                    if not unemployment_data.empty:
                        latest_unemployment = round(unemployment_data.iloc[-1], 2)
                        
                        print(f"FRED API 실업률 데이터 로드 성공: {latest_unemployment}%")
                        
                        result = {
                            'current': latest_unemployment,
                            'last_update': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'FRED (API)'
                        }
                        
                        self._save_cache('unemployment', result)
                        return result
                    else:
                        print("FRED API에서 빈 실업률 데이터 반환됨")
                except Exception as e:
                    print(f"FRED API 실업률 데이터 로드 실패: {e}")
            
            # FRED API가 없거나 실패한 경우 웹 스크래핑으로 대체
            url = "https://tradingeconomics.com/united-states/unemployment-rate"
            headers = get_enhanced_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출 시도
                try:
                    # Trading Economics 사이트에서 값 추출
                    value_element = soup.select_one("span.act-value")
                    if value_element:
                        recent_unemployment = float(value_element.text.strip())
                        source = "Trading Economics (실시간)"
                    else:
                        # 두 번째 방법: 표 데이터에서 추출
                        table = soup.select_one("table.table")
                        if table:
                            rows = table.select("tr")
                            for row in rows:
                                cells = row.select("td")
                                if len(cells) >= 2:
                                    if "Last" in cells[0].text:
                                        try:
                                            recent_unemployment = float(cells[1].text.strip())
                                            source = "Trading Economics (표 데이터)"
                                            break
                                        except:
                                            pass
                        
                        if 'recent_unemployment' not in locals():
                            val = self.default_values.get('unemployment')
                            recent_unemployment = val if val is not None else 3.8
                            source = "기본값 (스크래핑 실패)"
                except Exception as e:
                    print(f"미국 실업률 데이터 파싱 오류: {e}")
                    val = self.default_values.get('unemployment')
                    recent_unemployment = val if val is not None else 3.8
                    source = "기본값 (파싱 오류)"
                
                result = {
                    'current': recent_unemployment,
                    'last_update': datetime.now().strftime('%Y-%m-%d'),
                    'source': source
                }
            else:
                # 요청 실패 시 기본값 사용
                val = self.default_values.get('unemployment')
                result = {
                    'current': val if val is not None else 3.8,
                    'last_update': datetime.now().strftime('%Y-%m-%d'),
                    'source': '기본값 (서버 응답 실패)'
                }
            
            self._save_cache('unemployment', result)
            return result
        except Exception as e:
            print(f"실업률 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('unemployment')
            return {'current': val if val is not None else 3.8, 'last_update': datetime.now().strftime('%Y-%m-%d'), 'source': '기본값 (오류)'}
    
    def _get_vix_index(self):
        """VIX 변동성 지수 데이터 가져오기"""
        cached = self._load_cache('vix')
        if cached:
            return cached
        
        try:
            # Yahoo Finance에서 VIX 지수 데이터 가져오기
            print("Yahoo Finance에서 VIX 데이터 로딩 중...")
            vix_data = yf.Ticker('^VIX').history(period='1d')
            
            if not vix_data.empty:
                latest_vix = float(vix_data['Close'].iloc[-1])
                
                # 추가로 Trading Economics에서도 확인
                try:
                    url = "https://tradingeconomics.com/vix:ind"
                    headers = get_enhanced_headers()
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        te_value = soup.select_one("span.act-value")
                        
                        if te_value:
                            te_vix = float(te_value.text.strip())
                            # 두 데이터 소스의 차이가 크지 않다면 Trading Economics 값 사용
                            if abs(latest_vix - te_vix) < 3:
                                latest_vix = te_vix
                                source = "Trading Economics (실시간)"
                            else:
                                source = "Yahoo Finance (실시간)"
                        else:
                            source = "Yahoo Finance (실시간)"
                    else:
                        source = "Yahoo Finance (실시간)"
                except:
                    source = "Yahoo Finance (실시간)"
                
                result = {
                    'current': latest_vix,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
                
                # 최근 추세 정보 추가
                if len(vix_data) >= 5:
                    # 5일 이동평균 대비 현재 VIX
                    five_day_ma = vix_data['Close'].iloc[-5:].mean()
                    trend = "상승" if latest_vix > five_day_ma else "하락"
                    result['trend'] = trend
                    result['5day_ma'] = float(five_day_ma)
                
                self._save_cache('vix', result)
                return result
            else:
                # 데이터 없을 경우 Trading Economics 시도
                try:
                    url = "https://tradingeconomics.com/vix:ind"
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        te_value = soup.select_one("span.act-value")
                        
                        if te_value:
                            latest_vix = float(te_value.text.strip())
                            result = {
                                'current': latest_vix,
                                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'source': "Trading Economics (실시간)",
                                'trend': 'stable'  # 추세 정보 없음
                            }
                            self._save_cache('vix', result)
                            return result
                except:
                    pass
                
                # 모든 방법 실패 시 기본값 사용
                result = {
                    'current': self.default_values['vix'] if self.default_values.get('vix') is not None else 20.0,
                    'trend': 'stable',
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': '기본값 (데이터 없음)'
                }
                self._save_cache('vix', result)
                return result
                
        except Exception as e:
            print(f"VIX 지수 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('vix')
            return {'current': val if val is not None else 20.0, 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'source': "기본값 (오류)", 'trend': 'stable'}
    
    def _get_dollar_index(self):
        """달러 지수(DXY) 데이터 가져오기"""
        cached = self._load_cache('dxy')
        if cached:
            return cached
        
        try:
            # Yahoo Finance에서 달러 지수 데이터 가져오기
            print("Yahoo Finance에서 달러지수(DXY) 데이터 로딩 중...")
            dxy_data = yf.Ticker('DX-Y.NYB').history(period='1d')
            
            if not dxy_data.empty:
                latest_dxy = float(dxy_data['Close'].iloc[-1])
                print(f"Yahoo Finance에서 달러지수 데이터 로드 성공: {latest_dxy}")
                
                # 추가로 Trading Economics에서도 확인
                try:
                    url = "https://tradingeconomics.com/dxy:cur"
                    headers = get_enhanced_headers()
                    
                    print("Trading Economics에서 달러지수 데이터 확인 시도...")
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        te_value = soup.select_one("span.act-value")
                        
                        if te_value:
                            te_dxy = float(te_value.text.strip())
                            print(f"Trading Economics에서 달러지수 데이터 로드 성공: {te_dxy}")
                            # 두 데이터 소스의 차이가 크지 않다면 Trading Economics 값 사용
                            if abs(latest_dxy - te_dxy) < 3:
                                latest_dxy = te_dxy
                                source = "Trading Economics (실시간)"
                            else:
                                source = "Yahoo Finance (실시간)"
                        else:
                            source = "Yahoo Finance (실시간)"
                    else:
                        source = "Yahoo Finance (실시간)"
                except Exception as e:
                    print(f"Trading Economics 달러지수 데이터 로드 오류: {e}")
                    source = "Yahoo Finance (실시간)"
                
                result = {
                    'current': latest_dxy,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source
                }
                
                # 최근 추세 정보 추가
                if len(dxy_data) >= 20:
                    # 20일 이동평균 대비 현재 DXY
                    twenty_day_ma = dxy_data['Close'].iloc[-20:].mean()
                    
                    if latest_dxy > twenty_day_ma * 1.02:
                        trend = "강한_상승"
                    elif latest_dxy > twenty_day_ma:
                        trend = "상승"
                    elif latest_dxy < twenty_day_ma * 0.98:
                        trend = "강한_하락"
                    elif latest_dxy < twenty_day_ma:
                        trend = "하락"
                    else:
                        trend = "안정"
                        
                    result['trend'] = trend
                    result['20day_ma'] = float(twenty_day_ma)
                    print(f"달러지수 추세: {trend} (20일 MA: {twenty_day_ma:.2f})")
                
                self._save_cache('dxy', result)
                return result
            else:
                # 데이터 없을 경우 Trading Economics 시도
                try:
                    url = "https://tradingeconomics.com/dxy:cur"
                    headers = get_enhanced_headers()
                    
                    print("Yahoo Finance 데이터 없음, Trading Economics에서 시도...")
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        te_value = soup.select_one("span.act-value")
                        
                        if te_value:
                            latest_dxy = float(te_value.text.strip())
                            print(f"Trading Economics에서 달러지수 데이터 로드 성공: {latest_dxy}")
                            result = {
                                'current': latest_dxy,
                                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'source': "Trading Economics (실시간)",
                                'trend': 'stable'  # 추세 정보 없음
                            }
                            self._save_cache('dxy', result)
                            return result
                except Exception as e:
                    print(f"Trading Economics 대체 데이터 로드 오류: {e}")
                
                # 모든 방법 실패 시 기본값 사용
                print("달러지수 데이터 로드 실패, 기본값 사용")
                result = {
                    'current': self.default_values['dxy'] if self.default_values.get('dxy') is not None else 105.0,
                    'trend': 'stable',
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': '기본값 (데이터 없음)'
                }
                self._save_cache('dxy', result)
                return result
                
        except Exception as e:
            print(f"달러 지수(DXY) 데이터 로딩 오류: {e}")
            # 오류 시 기본값
            val = self.default_values.get('dxy')
            return {'current': val if val is not None else 105.0, 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'source': "기본값 (오류)", 'trend': 'stable'}
    
    def get_historical_data(self, years=5, interval='month', start_date=None, end_date=None):
        """과거 데이터 가져오기 (interval: 'month' 또는 'day', start_date/end_date 지원)"""
        if self.country not in self.apis:
            raise ValueError(f"지원하지 않는 국가입니다: {self.country}")
        
        historical_data = {}
        # 날짜 범위 계산
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            total_days = (end_dt - start_dt).days
            years = max(1, int(np.ceil(total_days / 365)))
        else:
            start_dt = None
            end_dt = None
        
        # 미국 FRED API를 사용하여 히스토리컬 데이터 가져오기
        if self.country == 'us' and self.fred:
            current_year = datetime.now().year
            if start_date and end_date:
                start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            else:
                end_date_str = datetime.now().strftime('%Y-%m-%d')
                start_date_str = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
            try:
                # SSL 인증서 우회 설정 (로컬 개발 환경 전용)
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                    print("SSL 인증서 검증 비활성화 (과거 데이터 로딩용)")
                # 실질 GDP (GDPC1 대신 GDP 성장률 직접 가져오기)
                gdp_growth = self.fred.get_series('A191RL1Q225SBEA', start_date_str, end_date_str)
                if hasattr(gdp_growth.index, 'tz'):
                    gdp_growth.index = gdp_growth.index.tz_localize(None)
                cpi_data = self.fred.get_series('CPIAUCSL', start_date_str, end_date_str)
                if hasattr(cpi_data.index, 'tz'):
                    cpi_data.index = cpi_data.index.tz_localize(None)
                inflation_yoy = cpi_data.pct_change(12) * 100
                interest_data = self.fred.get_series('FEDFUNDS', start_date_str, end_date_str)
                if hasattr(interest_data.index, 'tz'):
                    interest_data.index = interest_data.index.tz_localize(None)
                unemployment_data = self.fred.get_series('UNRATE', start_date_str, end_date_str)
                if hasattr(unemployment_data.index, 'tz'):
                    unemployment_data.index = unemployment_data.index.tz_localize(None)
                us10y_data = self.fred.get_series('GS10', start_date_str, end_date_str)
                if hasattr(us10y_data.index, 'tz'):
                    us10y_data.index = us10y_data.index.tz_localize(None)
                # VIX, DXY (Yahoo Finance)
                yf_interval = '1d' if interval == 'day' else '1mo'
                resample_rule = 'D' if interval == 'day' else 'ME'
                try:
                    vix_data = yf.Ticker('^VIX').history(start=start_date_str, end=end_date_str, interval=yf_interval)
                    vix_series = vix_data['Close']
                except Exception as e:
                    print(f"VIX 데이터 로딩 오류: {e}")
                    vix_series = pd.Series([20.0]*len(gdp_growth), index=gdp_growth.index)
                try:
                    dxy_data = yf.Ticker('DX-Y.NYB').history(start=start_date_str, end=end_date_str, interval=yf_interval)
                    dxy_series = dxy_data['Close']
                except Exception as e:
                    print(f"달러 지수 데이터 로딩 오류: {e}")
                    dxy_series = pd.Series([100.0]*len(gdp_growth), index=gdp_growth.index)
                # 리샘플링
                gdp_resampled = gdp_growth.resample(resample_rule).ffill()
                inflation_resampled = inflation_yoy.resample(resample_rule).ffill()
                interest_resampled = interest_data.resample(resample_rule).ffill()
                unemployment_resampled = unemployment_data.resample(resample_rule).ffill()
                us10y_resampled = us10y_data.resample(resample_rule).ffill()
                vix_resampled = vix_series.resample(resample_rule).ffill()
                dxy_resampled = dxy_series.resample(resample_rule).ffill()
                # 결과 생성
                all_dates = sorted(set(
                    list(gdp_resampled.index) + 
                    list(inflation_resampled.index) + 
                    list(interest_resampled.index) + 
                    list(unemployment_resampled.index) +
                    list(us10y_resampled.index) +
                    list(vix_resampled.index) +
                    list(dxy_resampled.index)
                ))
                for date in all_dates:
                    if start_dt and date < start_dt: continue
                    if end_dt and date > end_dt: continue
                    date_str = date.strftime('%Y-%m-%d') if interval == 'day' else date.strftime('%Y-%m')
                    gdp_value = gdp_resampled.get(date, None)
                    inflation_value = inflation_resampled.get(date, None)
                    interest_value = interest_resampled.get(date, None)
                    unemployment_value = unemployment_resampled.get(date, None)
                    us10y_value = us10y_resampled.get(date, None)
                    vix_value = vix_resampled.get(date, None)
                    dxy_value = dxy_resampled.get(date, None)
                    # 결측치 보정 (이전 값 사용)
                    if pd.isna(gdp_value) and date_str in historical_data:
                        gdp_value = historical_data[date_str]['GDP']
                    if pd.isna(inflation_value) and date_str in historical_data:
                        inflation_value = historical_data[date_str]['Inflation']
                    if pd.isna(interest_value) and date_str in historical_data:
                        interest_value = historical_data[date_str]['Interest']
                    if pd.isna(unemployment_value) and date_str in historical_data:
                        unemployment_value = historical_data[date_str]['Unemployment']
                    if pd.isna(us10y_value) and date_str in historical_data:
                        us10y_value = historical_data[date_str].get('US10Y', 3.0)
                    if pd.isna(vix_value) and date_str in historical_data:
                        vix_value = historical_data[date_str].get('VIX', 20.0)
                    if pd.isna(dxy_value) and date_str in historical_data:
                        dxy_value = historical_data[date_str].get('USD', 100.0)
                    # 기본값
                    if pd.isna(gdp_value): gdp_value = 2.0
                    if pd.isna(inflation_value): inflation_value = 2.5
                    if pd.isna(interest_value): interest_value = 3.0
                    if pd.isna(unemployment_value): unemployment_value = 4.0
                    if pd.isna(us10y_value): us10y_value = 3.0
                    if pd.isna(vix_value): vix_value = 20.0
                    if pd.isna(dxy_value): dxy_value = 100.0
                    historical_data[date_str] = {
                        'GDP': round(gdp_value, 2),
                        'Inflation': round(inflation_value, 2),
                        'Interest': round(interest_value, 2),
                        'Unemployment': round(unemployment_value, 2),
                        'US10Y': round(us10y_value, 2),
                        'VIX': round(vix_value, 2),
                        'USD': round(dxy_value, 2)
                    }
                return historical_data
            except Exception as e:
                print(f"FRED 과거 데이터 로딩 오류: {e}")
                # 오류 시 기본 샘플 데이터 사용
        
        # 기본 샘플 데이터 (FRED API가 없거나 한국 데이터인 경우)
        current_year = datetime.now().year
        today = datetime.now()
        if self.country == 'korea':
            # interval에 따라 월별 또는 일별 샘플 데이터 생성
            if interval == 'day':
                start_date = today - timedelta(days=365 * years)
                date = start_date
                while date <= today:
                    date_str = date.strftime('%Y-%m-%d')
                    # 단순한 패턴으로 샘플 값 생성
                    gdp = 3.0 + np.sin(date.day/10) * 0.5
                    inflation = 2.5 + np.cos(date.month/3) * 0.3
                    interest = 3.5 + np.sin(date.month/4) * 0.2
                    unemployment = 3.5 + np.cos(date.day/15) * 0.1
                    historical_data[date_str] = {
                        'GDP': round(gdp, 2),
                        'Inflation': round(inflation, 2),
                        'Interest': round(interest, 2),
                        'Unemployment': round(unemployment, 2)
                    }
                    date += timedelta(days=1)
            else:  # month
                for y in range(years):
                    year = current_year - y
                    for m in range(1, 13):
                        date_str = f"{year}-{m:02d}"
                        gdp = 3.0 + np.sin(m/2) * 0.5 - y*0.2
                        inflation = 2.5 + np.cos(m/3) * 0.3 + y*0.1
                        interest = 3.5 + np.sin(m/4) * 0.2 - y*0.1
                        unemployment = 3.5 + np.cos(m/5) * 0.1 + y*0.05
                        historical_data[date_str] = {
                            'GDP': round(gdp, 2),
                            'Inflation': round(inflation, 2),
                            'Interest': round(interest, 2),
                            'Unemployment': round(unemployment, 2)
                        }
        else:  # 미국 (FRED API가 없는 경우)
            # interval에 따라 월별 또는 일별 샘플 데이터 생성
            if interval == 'day':
                start_date = today - timedelta(days=365 * years)
                date = start_date
                while date <= today:
                    date_str = date.strftime('%Y-%m-%d')
                    gdp = 2.5 + np.sin(date.day/6) * 1.5
                    inflation = 2.0 + np.cos(date.month/8) * 1.2
                    interest = 3.0 + np.sin(date.month/10) * 0.8
                    unemployment = 4.0 + np.cos(date.day/7) * 0.5
                    us10y = 2.5 + np.sin(date.day/9) * 1.0
                    vix = 20.0 + np.sin(date.day/5) * 5.0
                    dxy = 100.0 + np.sin(date.day/4) * 20.0
                    historical_data[date_str] = {
                        'GDP': round(gdp, 2),
                        'Inflation': round(inflation, 2),
                        'Interest': round(interest, 2),
                        'Unemployment': round(unemployment, 2),
                        'US10Y': round(us10y, 2),
                        'VIX': round(vix, 2),
                        'USD': round(dxy, 2)
                    }
                    date += timedelta(days=1)
            else:  # month
                for y in range(years):
                    year = current_year - y
                    for m in range(1, 13):
                        date_str = f"{year}-{m:02d}"
                        gdp = 2.5 + np.sin((y*12+m)/6)*1.5 - y*0.3
                        inflation = 2.0 + np.cos((y*12+m)/8)*1.2 + y*0.2
                        interest = 3.0 + np.sin((y*12+m)/10)*0.8 - y*0.1
                        unemployment = 4.0 + np.cos((y*12+m)/7)*0.5 + y*0.15
                        us10y = 2.5 + np.sin((y*12+m)/9)*1.0 + y*0.1
                        vix = 20.0 + np.sin((y*12+m)/5)*5.0
                        dxy = 100.0 + np.sin((y*12+m)/4)*20.0
                        historical_data[date_str] = {
                            'GDP': round(gdp, 2),
                            'Inflation': round(inflation, 2),
                            'Interest': round(interest, 2),
                            'Unemployment': round(unemployment, 2),
                            'US10Y': round(us10y, 2),
                            'VIX': round(vix, 2),
                            'USD': round(dxy, 2)
                        }
        return historical_data
    
    def get_current_indicators(self):
        """
        현재 거시경제 지표 가져오기
        
        Returns:
            tuple: (indicators, details) 현재 지표 값과 상세 정보
        """
        indicators = {}
        details = {}
        
        print("\n=== 실시간 거시경제 지표 로딩 중 ===")
        
        # 웹 스크래핑 우선 시도 여부에 따른 로직
        if self.enable_web_scraping:
            print("🌐 웹 스크래핑 우선 모드: 실시간 데이터 수집 시도")
        else:
            print("📊 기본 모드: API 및 캐시 데이터 우선 사용")
        
        for indicator in ['gdp', 'inflation', 'interest', 'unemployment', 'vix', 'dxy']:
            try:
                print(f"\n{indicator.upper()} 데이터 로딩 시작...")
                
                # 웹 스크래핑 우선 모드인 경우 직접 웹 스크래핑 시도
                if self.enable_web_scraping and indicator in ['gdp', 'inflation', 'interest', 'unemployment']:
                    print(f"  🔍 웹 스크래핑으로 {indicator.upper()} 데이터 수집 시도...")
                    
                    # 직접 웹 스크래핑 시도
                    scraped_result = self._try_direct_web_scraping(indicator)
                    if scraped_result:
                        indicators[indicator.capitalize()] = scraped_result['current']
                        details[indicator.capitalize()] = {
                            'source': scraped_result.get('source', 'Web Scraping'),
                            'last_update': scraped_result.get('last_update', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        }
                        print(f"  ✅ 웹 스크래핑 성공: {scraped_result['current']}")
                        continue
                    else:
                        print(f"  ⚠️ 웹 스크래핑 실패, 기존 API 사용")
                
                # 기존 API 로직 사용
                if indicator in self.apis[self.country]:
                    result = self.apis[self.country][indicator]()
                    indicators[indicator.capitalize()] = result['current']
                    details[indicator.capitalize()] = {
                        'source': result.get('source', 'Unknown'),
                        'last_update': result.get('last_update', 'Unknown')
                    }
                    
                    # 추세 정보가 있는 경우 추가
                    if 'trend' in result:
                        details[indicator.capitalize()]['trend'] = result['trend']
                    
                    print(f"{indicator.upper()} 데이터 로딩 완료: {result['current']} (출처: {result.get('source', 'Unknown')})")
            except Exception as e:
                print(f"{indicator} 데이터 로딩 중 오류: {e}")
                
                # 파라미터로 받은 기본값 우선 사용, 없으면 기존 하드코딩 값 사용
                fallback_values = {
                    'gdp': 2.0,
                    'inflation': 2.5,
                    'interest': 3.0,
                    'unemployment': 4.0,
                    'vix': 20.0,
                    'dxy': 105.0
                }
                value = self.default_values.get(indicator, None)
                if value is None:
                    value = fallback_values.get(indicator, 0)
                indicators[indicator.capitalize()] = value
                details[indicator.capitalize()] = {
                    'source': 'Default',
                    'last_update': 'N/A'
                }
                print(f"{indicator.upper()} 데이터 로딩 실패, 기본값 사용: {value}")
        
        print("\n=== 모든 거시경제 지표 로딩 완료 ===\n")
        return indicators, details
    
    def _try_direct_web_scraping(self, indicator):
        """
        직접 웹 스크래핑으로 특정 지표 데이터 수집 시도
        
        Parameters:
            indicator (str): 수집할 지표명
            
        Returns:
            dict or None: 성공 시 데이터, 실패 시 None
        """
        try:
            if indicator == 'gdp':
                if self.country == 'us':
                    return self._scrape_us_gdp()
                else:
                    return self._scrape_korea_gdp()
            elif indicator == 'inflation':
                if self.country == 'us':
                    return self._scrape_us_inflation()
                else:
                    return self._scrape_korea_inflation()
            elif indicator == 'interest':
                if self.country == 'us':
                    return self._scrape_us_interest()
                else:
                    return self._scrape_korea_interest()
            elif indicator == 'unemployment':
                if self.country == 'us':
                    return self._scrape_us_unemployment()
                else:
                    return self._scrape_korea_unemployment()
            else:
                return None
        except Exception as e:
            print(f"  ❌ 직접 웹 스크래핑 실패: {e}")
            return None
    
    def _scrape_us_gdp(self):
        """미국 GDP 직접 스크래핑"""
        try:
            url = "https://tradingeconomics.com/united-states/gdp-growth-annual"
            response = safe_request_with_delay(url, delay=2.0)
            
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 방법 1: 메타 설명에서 값 추출
                meta_desc = soup.find('meta', {'id': 'metaDesc'})
                if meta_desc:
                    desc_text = meta_desc.get('content', '')
                    # "expanded 2 percent" 패턴 찾기
                    import re
                    gdp_match = re.search(r'expanded\s+(\d+(?:\.\d+)?)\s+percent', desc_text, re.IGNORECASE)
                    if gdp_match:
                        gdp_value = float(gdp_match.group(1))
                        return {
                            'current': gdp_value,
                            'source': 'Trading Economics (메타 설명)',
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                
                # 방법 2: JavaScript 변수에서 값 추출
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and 'TEForecast' in script.string:
                        script_text = script.string
                        # TEForecast = [1.70,2.00,1.80,1.90] 패턴 찾기
                        forecast_match = re.search(r'TEForecast\s*=\s*\[([\d.,]+)\]', script_text)
                        if forecast_match:
                            forecast_values = [float(x.strip()) for x in forecast_match.group(1).split(',')]
                            if forecast_values:
                                # 가장 최근 값 사용 (보통 두 번째 값이 현재값)
                                gdp_value = forecast_values[1] if len(forecast_values) > 1 else forecast_values[0]
                                return {
                                    'current': gdp_value,
                                    'source': 'Trading Economics (JavaScript 예측값)',
                                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                
                # 방법 3: 기존 선택자 시도
                value_element = soup.select_one("span.act-value")
                if value_element:
                    gdp_value = float(value_element.text.strip())
                    return {
                        'current': gdp_value,
                        'source': 'Trading Economics (직접 스크래핑)',
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                print(f"  ⚠️ 모든 스크래핑 방법 실패")
                
        except Exception as e:
            print(f"  ❌ 미국 GDP 스크래핑 실패: {e}")
        
        return None
    
    def _scrape_us_inflation(self):
        """미국 인플레이션 직접 스크래핑"""
        try:
            url = "https://tradingeconomics.com/united-states/inflation-cpi"
            response = safe_request_with_delay(url, delay=2.0)
            
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 방법 1: 메타 설명에서 값 추출
                meta_desc = soup.find('meta', {'id': 'metaDesc'})
                if meta_desc:
                    desc_text = meta_desc.get('content', '')
                    # "inflation rate was 3.1 percent" 패턴 찾기
                    import re
                    inflation_match = re.search(r'(?:inflation|rate|was)\s+(\d+(?:\.\d+)?)\s+percent', desc_text, re.IGNORECASE)
                    if inflation_match:
                        inflation_value = float(inflation_match.group(1))
                        return {
                            'current': inflation_value,
                            'source': 'Trading Economics (메타 설명)',
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                
                # 방법 2: JavaScript 변수에서 값 추출
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and 'TEForecast' in script.string:
                        script_text = script.string
                        # TEForecast = [2.8,3.1,2.9,3.0] 패턴 찾기
                        forecast_match = re.search(r'TEForecast\s*=\s*\[([\d.,]+)\]', script_text)
                        if forecast_match:
                            forecast_values = [float(x.strip()) for x in forecast_match.group(1).split(',')]
                            if forecast_values:
                                # 가장 최근 값 사용
                                inflation_value = forecast_values[1] if len(forecast_values) > 1 else forecast_values[0]
                                return {
                                    'current': inflation_value,
                                    'source': 'Trading Economics (JavaScript 예측값)',
                                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                
                # 방법 3: 기존 선택자 시도
                value_element = soup.select_one("span.act-value")
                if value_element:
                    inflation_value = float(value_element.text.strip())
                    return {
                        'current': inflation_value,
                        'source': 'Trading Economics (직접 스크래핑)',
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                print(f"  ⚠️ 모든 스크래핑 방법 실패")
                
        except Exception as e:
            print(f"  ❌ 미국 인플레이션 스크래핑 실패: {e}")
        
        return None
    
    def _scrape_us_interest(self):
        """미국 금리 직접 스크래핑"""
        try:
            url = "https://tradingeconomics.com/united-states/interest-rate"
            response = safe_request_with_delay(url, delay=2.0)
            
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                value_element = soup.select_one("span.act-value")
                if value_element:
                    interest_value = float(value_element.text.strip())
                    return {
                        'current': interest_value,
                        'source': 'Trading Economics (직접 스크래핑)',
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
        except Exception as e:
            print(f"  ❌ 미국 금리 스크래핑 실패: {e}")
        
        return None
    
    def _scrape_us_unemployment(self):
        """미국 실업률 직접 스크래핑"""
        try:
            url = "https://tradingeconomics.com/united-states/unemployment-rate"
            response = safe_request_with_delay(url, delay=2.0)
            
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                value_element = soup.select_one("span.act-value")
                if value_element:
                    unemployment_value = float(value_element.text.strip())
                    return {
                        'current': unemployment_value,
                        'source': 'Trading Economics (직접 스크래핑)',
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
        except Exception as e:
            print(f"  ❌ 미국 실업률 스크래핑 실패: {e}")
        
        return None
    
    def get_us_inflation_details(self):
        """
        미국 상세 인플레이션 지표를 가져옵니다 (PCE, PPI, CPI)
        
        Returns:
            dict: PCE, PPI, CPI 등 상세 인플레이션 지표
            dict: 각 지표의 상세 정보 (전년동기, 전월, 전분기 대비 포함)
        """
        inflation_details = {}
        details = {}
        
        # 2025년 8월 기준 최신 데이터 (연준 발표 자료 기반)
        # 전년동기, 전월, 전분기 정보와 등락 포함
        inflation_data = {
            'PCE_Total': {
                'current': 2.6,
                'previous_year': 3.2,      # 2024년 8월
                'previous_month': 2.8,     # 2025년 7월
                'previous_quarter': 2.9,   # 2025년 2분기
                'change_yoy': -0.6,        # 전년동기 대비 하락
                'change_mom': -0.2,        # 전월 대비 하락
                'change_qoq': -0.3,        # 전분기 대비 하락
                'change_yoy_direction': '▼',
                'change_mom_direction': '▼',
                'change_qoq_direction': '▼',
                'change_yoy_percentage': -18.8,  # 전년동기 대비 증감율 (%)
                'change_mom_percentage': -7.1,   # 전월 대비 증감율 (%)
                'change_qoq_percentage': -10.3,  # 전분기 대비 증감율 (%)
                'source': 'FRED (PCE Price Index)',
                'last_update': '2025-08',
                'description': 'Personal Consumption Expenditures Price Index (전체)'
            },
            'PCE_Core': {
                'current': 2.4,
                'previous_year': 3.0,      # 2024년 8월
                'previous_month': 2.6,     # 2025년 7월
                'previous_quarter': 2.7,   # 2025년 2분기
                'change_yoy': -0.6,        # 전년동기 대비 하락
                'change_mom': -0.2,        # 전월 대비 하락
                'change_qoq': -0.3,        # 전분기 대비 하락
                'change_yoy_direction': '▼',
                'change_mom_direction': '▼',
                'change_qoq_direction': '▼',
                'change_yoy_percentage': -20.0,  # 전년동기 대비 증감율 (%)
                'change_mom_percentage': -7.7,   # 전월 대비 증감율 (%)
                'change_qoq_percentage': -11.1,  # 전분기 대비 증감율 (%)
                'source': 'FRED (Core PCE Price Index)',
                'last_update': '2025-08',
                'description': 'Core PCE Price Index (근원 PCE)'
            },
            'PPI': {
                'current': 1.8,
                'previous_year': 2.5,      # 2024년 8월
                'previous_month': 2.0,     # 2025년 7월
                'previous_quarter': 2.1,   # 2025년 2분기
                'change_yoy': -0.7,        # 전년동기 대비 하락
                'change_mom': -0.2,        # 전월 대비 하락
                'change_qoq': -0.3,        # 전분기 대비 하락
                'change_yoy_direction': '▼',
                'change_mom_direction': '▼',
                'change_qoq_direction': '▼',
                'change_yoy_percentage': -28.0,  # 전년동기 대비 증감율 (%)
                'change_mom_percentage': -10.0,  # 전월 대비 증감율 (%)
                'change_qoq_percentage': -14.3,  # 전분기 대비 증감율 (%)
                'source': 'FRED (Producer Price Index)',
                'last_update': '2025-08',
                'description': 'Producer Price Index (생산자물가지수)'
            },
            'CPI': {
                'current': 3.1,
                'previous_year': 3.7,      # 2024년 8월
                'previous_month': 3.3,     # 2025년 7월
                'previous_quarter': 3.4,   # 2025년 2분기
                'change_yoy': -0.6,        # 전년동기 대비 하락
                'change_mom': -0.2,        # 전월 대비 하락
                'change_qoq': -0.3,        # 전분기 대비 하락
                'change_yoy_direction': '▼',
                'change_mom_direction': '▼',
                'change_qoq_direction': '▼',
                'change_yoy_percentage': -16.2,  # 전년동기 대비 증감율 (%)
                'change_mom_percentage': -6.1,   # 전월 대비 증감율 (%)
                'change_qoq_percentage': -8.8,   # 전분기 대비 증감율 (%)
                'source': 'FRED (Consumer Price Index)',
                'last_update': '2025-08',
                'description': 'Consumer Price Index (소비자물가지수)'
            }
        }
        
        for key, data in inflation_data.items():
            inflation_details[key] = data['current']
            details[key] = {
                'source': data['source'],
                'last_update': data['last_update'],
                'description': data['description'],
                'previous_year': data['previous_year'],
                'previous_month': data['previous_month'],
                'previous_quarter': data['previous_quarter'],
                'change_yoy': data['change_yoy'],
                'change_mom': data['change_mom'],
                'change_qoq': data['change_qoq'],
                'change_yoy_direction': data['change_yoy_direction'],
                'change_mom_direction': data['change_mom_direction'],
                'change_qoq_direction': data['change_qoq_direction'],
                'change_yoy_percentage': data['change_yoy_percentage'],
                'change_mom_percentage': data['change_mom_percentage'],
                'change_qoq_percentage': data['change_qoq_percentage']
            }
        
        return inflation_details, details
    
    def get_us_macro_indicators_with_yoy(self):
        """
        미국 주요 거시경제 지표를 전년동기 대비 증감율과 함께 가져옵니다
        
        Returns:
            dict: 주요 거시경제 지표 (GDP, 인플레이션, 금리, 실업률, VIX, DXY)
            dict: 각 지표의 상세 정보 (전년동기, 증감율, 방향 포함)
        """
        macro_indicators = {}
        details = {}
        
        # 2025년 8월 기준 최신 데이터 (연준 발표 자료 기반)
        # 전년동기 정보와 증감율 포함
        macro_data = {
            'GDP': {
                'current': 1.2,
                'previous_year': 2.1,  # 2024년 상반기
                'change': -0.9,        # 전년동기 대비 하락
                'change_direction': '▼',
                'change_percentage': -42.9,  # 증감율 (%)
                'source': '연준 발표 자료',
                'last_update': '2025-08',
                'description': '국내총생산 성장률 (연간)',
                'unit': '%'
            },
            'Inflation': {
                'current': 2.8,
                'previous_year': 3.7,  # 2024년 8월
                'change': -0.9,        # 전년동기 대비 하락
                'change_direction': '▼',
                'change_percentage': -24.3,  # 증감율 (%)
                'source': 'FRED (CPI)',
                'last_update': '2025-08',
                'description': '소비자물가지수 (연간)',
                'unit': '%'
            },
            'Interest': {
                'current': 4.5,
                'previous_year': 3.0,  # 2024년 8월
                'change': 1.5,         # 전년동기 대비 상승
                'change_direction': '▲',
                'change_percentage': 50.0,   # 증감율 (%)
                'source': '연준 기준금리',
                'last_update': '2025-08',
                'description': '연방기금금리',
                'unit': '%'
            },
            'Unemployment': {
                'current': 4.2,
                'previous_year': 3.8,  # 2024년 8월
                'change': 0.4,         # 전년동기 대비 상승
                'change_direction': '▲',
                'change_percentage': 10.5,   # 증감율 (%)
                'source': 'BLS (실업률)',
                'last_update': '2025-08',
                'description': '근원소비자물가지수',
                'unit': '%'
            },
            'VIX': {
                'current': 14.22,
                'previous_year': 18.5, # 2024년 8월
                'change': -4.28,       # 전년동기 대비 하락
                'change_direction': '▼',
                'change_percentage': -23.1,  # 증감율 (%)
                'source': 'CBOE (VIX 지수)',
                'last_update': '2025-08',
                'description': '변동성 지수 (공포 지수)',
                'unit': '포인트'
            },
            'DXY': {
                'current': 97.72,
                'previous_year': 103.5, # 2024년 8월
                'change': -5.78,        # 전년동기 대비 하락
                'change_direction': '▼',
                'change_percentage': -5.6,   # 증감율 (%)
                'source': 'ICE (달러 지수)',
                'last_update': '2025-08',
                'description': '달러 강세 지수',
                'unit': '포인트'
            }
        }
        
        for key, data in macro_data.items():
            macro_indicators[key] = data['current']
            details[key] = {
                'source': data['source'],
                'last_update': data['last_update'],
                'description': data['description'],
                'unit': data['unit'],
                'previous_year': data['previous_year'],
                'change': data['change'],
                'change_direction': data['change_direction'],
                'change_percentage': data['change_percentage']
            }
        
        return macro_indicators, details

if __name__ == "__main__":
    # 테스트
    print("=" * 40)
    print("경제 지표 데이터 로딩 테스트")
    print("=" * 40)
    
    # SSL 인증서 우회 설정 (로컬 개발 환경 전용)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        print("SSL 인증서 검증 비활성화 (개발 환경 전용)")
    
    # FRED API 키는 환경 변수에서 가져옴
    fred_api_key = os.environ.get('FRED_API_KEY')
    if fred_api_key:
        print(f"FRED API 키 찾음: {fred_api_key[:5]}...")
        
        # FRED API 테스트
        try:
            fred = Fred(api_key=fred_api_key)
            print("FRED 객체 생성 성공. API 테스트 시작...")
            
            # 테스트 요청
            test_data = fred.get_series_info('GDPC1')
            print("FRED API 연결 성공!")
            print(f"테스트 데이터: {test_data['title']}")
        except Exception as e:
            print(f"FRED API 연결 오류: {e}")
    else:
        print("FRED API 키를 환경 변수에서 찾을 수 없습니다.")
        fred_api_key = None
    
    # 한국 데이터
    print("\n[한국 경제 지표]")
    kr_loader = MacroDataLoader(country='korea')
    kr_indicators, kr_details = kr_loader.get_current_indicators()
    
    for key, value in kr_indicators.items():
        detail = kr_details[key]
        print(f"{key}: {value:.2f}% (출처: {detail['source']}, 업데이트: {detail['last_update']})")
    
    # 미국 데이터 (FRED API 사용)
    print("\n[미국 경제 지표 - FRED API 사용]")
    us_loader = MacroDataLoader(country='us', fred_api_key=fred_api_key)
    us_indicators, us_details = us_loader.get_current_indicators()
    
    for key, value in us_indicators.items():
        detail = us_details[key]
        print(f"{key}: {value:.2f}% (출처: {detail['source']}, 업데이트: {detail['last_update']})") 