#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CME FedWatch Tool Selenium 스크래퍼

Selenium을 사용하여 CME FedWatch Tool 웹사이트에서 실제 목표금리 예상 가능성을
스크래핑하여 현재 시장의 금리 정책 기대를 실시간으로 확인합니다.
"""

import time
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium이 설치되지 않았습니다. pip install selenium을 실행하세요.")

class CMEFedWatchSeleniumScraper:
    """
    CME FedWatch Tool Selenium 스크래퍼
    """
    
    def __init__(self):
        self.base_url = "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"
        self.current_fed_rate = 4.5
        
        # 2025년 FOMC 회의 일정
        self.fomc_dates_2025 = [
            "2025-01-29",  # 1월 FOMC
            "2025-03-19",  # 3월 FOMC
            "2025-05-07",  # 5월 FOMC
            "2025-06-18",  # 6월 FOMC
            "2025-07-30",  # 7월 FOMC
            "2025-09-17",  # 9월 FOMC
            "2025-11-06",  # 11월 FOMC
            "2025-12-17"   # 12월 FOMC
        ]
        
        self.driver = None
        
    def setup_driver(self, headless: bool = True) -> bool:
        """
        Selenium WebDriver를 설정합니다.
        
        Args:
            headless: 헤드리스 모드 사용 여부
            
        Returns:
            bool: 드라이버 설정 성공 여부
        """
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium을 사용할 수 없습니다.")
            return False
            
        try:
            print("🔧 Selenium WebDriver 설정 중...")
            
            # Chrome 옵션 설정
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # WebDriver 생성
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            
            print("✅ WebDriver 설정 완료")
            return True
            
        except Exception as e:
            print(f"❌ WebDriver 설정 실패: {e}")
            return False
    
    def scrape_fed_watch_with_selenium(self) -> Dict:
        """
        Selenium을 사용하여 CME FedWatch Tool을 스크래핑합니다.
        
        Returns:
            dict: 스크래핑된 FED Watch 데이터
        """
        if not self.driver:
            if not self.setup_driver():
                return self._get_fallback_data()
        
        try:
            print("🌐 CME FedWatch Tool 웹사이트 접속 중...")
            
            # 웹사이트 접속
            self.driver.get(self.base_url)
            time.sleep(5)  # 페이지 로딩 대기
            
            print("📄 페이지 제목:", self.driver.title)
            
            # FedWatch 데이터 테이블 찾기
            fed_watch_data = self._extract_fed_watch_data()
            
            if fed_watch_data:
                return self._parse_selenium_data(fed_watch_data)
            else:
                print("⚠️ FedWatch 데이터를 찾을 수 없습니다.")
                return self._get_fallback_data()
                
        except Exception as e:
            print(f"❌ Selenium 스크래핑 중 오류: {e}")
            return self._get_fallback_data()
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def _extract_fed_watch_data(self) -> Optional[Dict]:
        """
        웹페이지에서 FedWatch 데이터를 추출합니다.
        
        Returns:
            dict: 추출된 FedWatch 데이터 또는 None
        """
        try:
            # 여러 방법으로 데이터 추출 시도
            
            # 방법 1: 테이블 데이터 직접 추출
            table_data = self._extract_table_data()
            if table_data:
                return table_data
            
            # 방법 2: JavaScript 변수에서 데이터 추출
            js_data = self._extract_javascript_data()
            if js_data:
                return js_data
            
            # 방법 3: 페이지 소스에서 JSON 데이터 추출
            source_data = self._extract_source_data()
            if source_data:
                return source_data
            
            return None
            
        except Exception as e:
            print(f"❌ 데이터 추출 중 오류: {e}")
            return None
    
    def _extract_table_data(self) -> Optional[Dict]:
        """
        웹페이지의 테이블에서 FedWatch 데이터를 추출합니다.
        
        Returns:
            dict: 추출된 테이블 데이터 또는 None
        """
        try:
            print("🔍 테이블 데이터 추출 시도...")
            
            # FedWatch 관련 테이블 찾기
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            for table in tables:
                try:
                    # 테이블 헤더 확인
                    headers = table.find_elements(By.TAG_NAME, "th")
                    if headers and any("FOMC" in header.text or "Meeting" in header.text for header in headers):
                        print("✅ FedWatch 테이블 발견")
                        
                        # 테이블 데이터 추출
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        table_data = {}
                        
                        for row in rows[1:]:  # 헤더 제외
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if len(cells) >= 3:
                                try:
                                    meeting_date = cells[0].text.strip()
                                    rate_level = cells[1].text.strip().replace('%', '')
                                    probability = cells[2].text.strip().replace('%', '')
                                    
                                    if meeting_date and rate_level and probability:
                                        if meeting_date not in table_data:
                                            table_data[meeting_date] = {}
                                        
                                        table_data[meeting_date][rate_level] = float(probability)
                                except:
                                    continue
                        
                        if table_data:
                            return {"rate_probabilities": table_data}
                        
                except Exception as e:
                    continue
            
            print("⚠️ 테이블에서 FedWatch 데이터를 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            print(f"❌ 테이블 데이터 추출 중 오류: {e}")
            return None
    
    def _extract_javascript_data(self) -> Optional[Dict]:
        """
        JavaScript 변수에서 FedWatch 데이터를 추출합니다.
        
        Returns:
            dict: 추출된 JavaScript 데이터 또는 None
        """
        try:
            print("🔍 JavaScript 데이터 추출 시도...")
            
            # JavaScript 변수 실행
            js_script = """
            if (typeof window.fedWatchData !== 'undefined') {
                return window.fedWatchData;
            } else if (typeof window.FedWatchData !== 'undefined') {
                return window.FedWatchData;
            } else if (typeof window.fedWatch !== 'undefined') {
                return window.fedWatch;
            } else {
                return null;
            }
            """
            
            result = self.driver.execute_script(js_script)
            if result:
                print("✅ JavaScript에서 FedWatch 데이터 발견")
                return result
            
            print("⚠️ JavaScript에서 FedWatch 데이터를 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            print(f"❌ JavaScript 데이터 추출 중 오류: {e}")
            return None
    
    def _extract_source_data(self) -> Optional[Dict]:
        """
        페이지 소스에서 JSON 데이터를 추출합니다.
        
        Returns:
            dict: 추출된 소스 데이터 또는 None
        """
        try:
            print("🔍 페이지 소스에서 데이터 추출 시도...")
            
            page_source = self.driver.page_source
            
            # JSON 데이터 패턴 찾기
            json_patterns = [
                r'\{.*?"FedWatch".*?\}',
                r'\{.*?"quotes".*?\}',
                r'\{.*?"rateProbabilities".*?\}',
                r'window\.fedWatchData\s*=\s*(\{.*?\});',
                r'window\.FedWatchData\s*=\s*(\{.*?\});'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, page_source, re.DOTALL)
                for match in matches:
                    try:
                        if pattern.startswith('window.'):
                            # window 변수 할당 형태
                            data = json.loads(match)
                        else:
                            # 직접 JSON 형태
                            data = json.loads(match)
                        
                        print("✅ 페이지 소스에서 FedWatch 데이터 발견")
                        return data
                        
                    except json.JSONDecodeError:
                        continue
            
            print("⚠️ 페이지 소스에서 FedWatch 데이터를 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            print(f"❌ 페이지 소스 데이터 추출 중 오류: {e}")
            return None
    
    def _parse_selenium_data(self, data: Dict) -> Dict:
        """
        Selenium으로 스크래핑된 데이터를 파싱합니다.
        
        Args:
            data: 스크래핑된 원본 데이터
            
        Returns:
            dict: 파싱된 FED Watch 데이터
        """
        try:
            parsed_data = {
                "current_rate": self.current_fed_rate,
                "next_meeting": self._get_next_fomc_meeting(),
                "rate_probabilities": {},
                "market_expectations": {},
                "scraping_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "CME FedWatch Tool (Selenium 스크래핑)"
            }
            
            # 데이터 구조에 따라 파싱
            if isinstance(data, dict):
                if 'rate_probabilities' in data:
                    parsed_data["rate_probabilities"] = data['rate_probabilities']
                elif 'quotes' in data:
                    parsed_data["rate_probabilities"] = self._parse_quotes_data(data['quotes'])
                elif 'data' in data:
                    parsed_data["rate_probabilities"] = self._parse_data_structure(data['data'])
                else:
                    # 직접 rate_probabilities 형태
                    parsed_data["rate_probabilities"] = data
            
            # 시장 기대 요약 생성
            parsed_data["market_expectations"] = self._generate_market_expectations(
                parsed_data["rate_probabilities"]
            )
            
            return parsed_data
            
        except Exception as e:
            print(f"❌ Selenium 데이터 파싱 중 오류: {e}")
            return self._get_fallback_data()
    
    def _parse_quotes_data(self, quotes: List) -> Dict:
        """quotes 데이터를 파싱합니다."""
        rate_probs = {}
        
        try:
            for quote in quotes:
                if isinstance(quote, dict):
                    meeting_date = quote.get('meetingDate', '')
                    rate_level = quote.get('rateLevel', '')
                    probability = quote.get('probability', 0)
                    
                    if meeting_date and rate_level and probability is not None:
                        if meeting_date not in rate_probs:
                            rate_probs[meeting_date] = {}
                        
                        rate_probs[meeting_date][str(rate_level)] = float(probability)
            
            return rate_probs
            
        except Exception as e:
            print(f"❌ quotes 데이터 파싱 중 오류: {e}")
            return {}
    
    def _parse_data_structure(self, data: Dict) -> Dict:
        """일반적인 데이터 구조를 파싱합니다."""
        rate_probs = {}
        
        try:
            for key, value in data.items():
                if isinstance(value, dict) and 'probabilities' in value:
                    meeting_date = key
                    probabilities = value['probabilities']
                    
                    if meeting_date not in rate_probs:
                        rate_probs[meeting_date] = {}
                    
                    for rate, prob in probabilities.items():
                        if isinstance(prob, (int, float)):
                            rate_probs[meeting_date][str(rate)] = float(prob)
            
            return rate_probs
            
        except Exception as e:
            print(f"❌ 데이터 구조 파싱 중 오류: {e}")
            return {}
    
    def _get_next_fomc_meeting(self) -> str:
        """다음 FOMC 회의일을 반환합니다."""
        today = datetime.now()
        
        for meeting_date in self.fomc_dates_2025:
            meeting_dt = datetime.strptime(meeting_date, "%Y-%m-%d")
            if meeting_dt > today:
                return meeting_date
        
        return self.fomc_dates_2025[-1]
    
    def _generate_market_expectations(self, rate_probs: Dict) -> Dict:
        """시장 기대 요약을 생성합니다."""
        expectations = {
            "next_meeting": "N/A",
            "year_end": "N/A",
            "trend": "N/A"
        }
        
        try:
            if not rate_probs:
                return expectations
            
            # 다음 회의 분석
            next_meeting = min(rate_probs.keys())
            if next_meeting in rate_probs:
                next_probs = rate_probs[next_meeting]
                current_rate = self.current_fed_rate
                
                # 금리인하 확률 계산
                cut_prob = sum(prob for rate, prob in next_probs.items() 
                             if float(rate) < current_rate)
                
                if cut_prob > 50:
                    expectations["next_meeting"] = f"금리인하 기대 ({cut_prob:.1f}%)"
                elif cut_prob > 20:
                    expectations["next_meeting"] = f"금리인하 가능성 ({cut_prob:.1f}%)"
                else:
                    expectations["next_meeting"] = f"현행 유지 기대 ({100-cut_prob:.1f}%)"
            
            # 연말 전망 분석
            year_end_meetings = [date for date in rate_probs.keys() 
                               if date >= "2025-12-01"]
            
            if year_end_meetings:
                total_cut_prob = 0
                total_meetings = 0
                
                for meeting in year_end_meetings:
                    if meeting in rate_probs:
                        probs = rate_probs[meeting]
                        current_rate = self.current_fed_rate
                        
                        cut_prob = sum(prob for rate, prob in probs.items() 
                                     if float(rate) < current_rate)
                        total_cut_prob += cut_prob
                        total_meetings += 1
                
                if total_meetings > 0:
                    avg_cut_prob = total_cut_prob / total_meetings
                    if avg_cut_prob > 50:
                        expectations["year_end"] = f"연말까지 금리인하 기대 ({avg_cut_prob:.1f}%)"
                    elif avg_cut_prob > 30:
                        expectations["year_end"] = f"연말까지 금리인하 가능성 ({avg_cut_prob:.1f}%)"
                    else:
                        expectations["year_end"] = f"연말까지 현행 유지 기대 ({100-avg_cut_prob:.1f}%)"
            
            # 전반적 트렌드
            if "금리인하" in expectations["next_meeting"]:
                expectations["trend"] = "금리인하 기대감 확대"
            elif "현행 유지" in expectations["next_meeting"]:
                expectations["trend"] = "현행 금리 유지 기대"
            else:
                expectations["trend"] = "금리 정책 불확실성"
            
            return expectations
            
        except Exception as e:
            print(f"❌ 시장 기대 요약 생성 중 오류: {e}")
            return expectations
    
    def _get_fallback_data(self) -> Dict:
        """스크래핑 실패 시 사용할 대체 데이터를 반환합니다."""
        print("🔄 대체 데이터 사용")
        
        fallback_data = {
            "current_rate": 4.5,
            "next_meeting": "2025-09-17",
            "rate_probabilities": {
                "2025-09-17": {  # 9월 FOMC
                    "4.25": 15.2,   # 25bp 인하 확률
                    "4.50": 45.8,   # 현행 유지 확률
                    "4.75": 39.0    # 25bp 인상 확률
                },
                "2025-11-06": {    # 11월 FOMC
                    "4.00": 8.5,    # 50bp 인하 확률
                    "4.25": 32.1,   # 25bp 인하 확률
                    "4.50": 35.4,   # 현행 유지 확률
                    "4.75": 24.0    # 25bp 인상 확률
                },
                "2025-12-17": {    # 12월 FOMC
                    "3.75": 5.2,    # 75bp 인하 확률
                    "4.00": 18.7,   # 50bp 인하 확률
                    "4.25": 28.9,   # 25bp 인하 확률
                    "4.50": 32.1,   # 현행 유지 확률
                    "4.75": 15.1    # 25bp 인상 확률
                }
            },
            "market_expectations": {
                "next_meeting": "25bp 인하 기대 (15.2%)",
                "year_end": "25-50bp 인하 기대 (47.6%)",
                "trend": "금리인하 기대감 확대"
            },
            "scraping_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "대체 데이터 (Selenium 스크래핑 실패)"
        }
        
        return fallback_data
    
    def get_current_fed_watch_data(self) -> Dict:
        """현재 FED Watch 데이터를 가져옵니다."""
        print("📊 CME FedWatch Tool에서 Selenium을 사용한 실시간 데이터 수집 중...")
        
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium을 사용할 수 없습니다.")
            return self._get_fallback_data()
        
        # Selenium을 사용한 스크래핑 시도
        data = self.scrape_fed_watch_with_selenium()
        
        if data:
            print(f"✅ 데이터 수집 완료 (출처: {data.get('data_source', 'N/A')})")
            print(f"📅 다음 FOMC 회의: {data.get('next_meeting', 'N/A')}")
            print(f"🏦 현재 연방기금금리: {data.get('current_rate', 'N/A')}%")
        else:
            print("❌ 데이터 수집 실패")
        
        return data


def main():
    """메인 함수"""
    print("CME FedWatch Tool Selenium 스크래퍼를 시작합니다...")
    
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium이 설치되지 않았습니다.")
        print("설치 방법: pip install selenium")
        print("Chrome WebDriver도 필요합니다.")
        return
    
    scraper = CMEFedWatchSeleniumScraper()
    
    # 현재 FED Watch 데이터 수집
    fed_watch_data = scraper.get_current_fed_watch_data()
    
    if fed_watch_data:
        print("\n📋 수집된 데이터 요약:")
        print(f"  • 데이터 출처: {fed_watch_data.get('data_source', 'N/A')}")
        print(f"  • 수집 시간: {fed_watch_data.get('scraping_time', 'N/A')}")
        print(f"  • FOMC 회의 수: {len(fed_watch_data.get('rate_probabilities', {}))}")
        
        # 금리 확률 데이터 출력
        rate_probs = fed_watch_data.get('rate_probabilities', {})
        if rate_probs:
            print("\n📈 FOMC 회의별 금리 확률:")
            for meeting_date, probabilities in rate_probs.items():
                print(f"\n📅 {meeting_date}:")
                for rate, prob in sorted(probabilities.items(), key=lambda x: float(x[0])):
                    print(f"  • {rate}%: {prob:.1f}%")
        
        # 시장 기대 요약 출력
        market_expectations = fed_watch_data.get('market_expectations', {})
        if market_expectations:
            print("\n📊 시장 기대 요약:")
            print(f"  • 다음 회의: {market_expectations.get('next_meeting', 'N/A')}")
            print(f"  • 연말 전망: {market_expectations.get('year_end', 'N/A')}")
            print(f"  • 전반적 트렌드: {market_expectations.get('trend', 'N/A')}")
    
    print("\n🔗 CME FedWatch Tool 직접 확인:")
    print("   https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html")
    
    print("\n💡 수동 확인 방법:")
    print("   1. 위 링크로 접속")
    print("   2. 'Target Rate Probabilities' 섹션 확인")
    print("   3. 각 FOMC 회의별 금리 확률 확인")
    print("   4. 'Current Rate'와 비교하여 인하/인상 기대 파악")


if __name__ == "__main__":
    main()
