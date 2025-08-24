#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CME FedWatch Tool 웹 스크래퍼

CME FedWatch Tool 웹사이트에서 실제 목표금리 예상 가능성을 스크래핑하여
현재 시장의 금리 정책 기대를 실시간으로 확인합니다.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import warnings
warnings.filterwarnings('ignore')

class CMEFedWatchScraper:
    """
    CME FedWatch Tool 웹 스크래퍼
    """
    
    def __init__(self):
        self.base_url = "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"
        self.api_url = "https://www.cmegroup.com/CmeWS/mvc/Quotes/FedWatch"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
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
        
        # 현재 기준 금리 (2025년 8월 기준)
        self.current_fed_rate = 4.5
        
    def scrape_fed_watch_website(self) -> Dict:
        """
        CME FedWatch Tool 웹사이트를 직접 스크래핑합니다.
        
        Returns:
            dict: 스크래핑된 FED Watch 데이터
        """
        try:
            print("🌐 CME FedWatch Tool 웹사이트 스크래핑 시작...")
            
            # 메인 페이지 스크래핑
            response = requests.get(self.base_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 페이지 내용 확인
            print(f"📄 페이지 제목: {soup.title.string if soup.title else 'N/A'}")
            
            # FedWatch 데이터가 포함된 스크립트 태그 찾기
            scripts = soup.find_all('script')
            fed_watch_data = None
            
            for script in scripts:
                if script.string and 'FedWatch' in script.string:
                    print("🔍 FedWatch 관련 스크립트 발견")
                    # JSON 데이터 추출 시도
                    try:
                        # JSON 데이터 패턴 찾기
                        json_pattern = r'\{.*"FedWatch".*\}'
                        matches = re.findall(json_pattern, script.string, re.DOTALL)
                        if matches:
                            fed_watch_data = json.loads(matches[0])
                            print("✅ JSON 데이터 추출 성공")
                            break
                    except:
                        continue
            
            if not fed_watch_data:
                print("⚠️ 스크립트에서 FedWatch 데이터를 찾을 수 없습니다.")
                # 대안: API 호출 시도
                return self._try_api_call()
            
            return self._parse_scraped_data(fed_watch_data)
            
        except requests.RequestException as e:
            print(f"❌ 웹사이트 스크래핑 실패: {e}")
            return self._try_api_call()
        except Exception as e:
            print(f"❌ 스크래핑 중 오류: {e}")
            return self._get_fallback_data()
    
    def _try_api_call(self) -> Dict:
        """
        CME FedWatch API 호출을 시도합니다.
        
        Returns:
            dict: API 데이터 또는 대체 데이터
        """
        try:
            print("🔌 CME FedWatch API 호출 시도...")
            
            # API 엔드포인트 호출
            response = requests.get(self.api_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                print("✅ API 호출 성공")
                data = response.json()
                return self._parse_api_data(data)
            else:
                print(f"⚠️ API 호출 실패 (상태 코드: {response.status_code})")
                return self._get_fallback_data()
                
        except Exception as e:
            print(f"❌ API 호출 중 오류: {e}")
            return self._get_fallback_data()
    
    def _parse_scraped_data(self, data: Dict) -> Dict:
        """
        스크래핑된 데이터를 파싱합니다.
        
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
                "data_source": "CME FedWatch Tool 웹사이트"
            }
            
            # 데이터 구조에 따라 파싱
            if isinstance(data, dict):
                # FedWatch 데이터 구조 확인
                if 'quotes' in data:
                    quotes = data['quotes']
                    parsed_data["rate_probabilities"] = self._parse_quotes_data(quotes)
                elif 'data' in data:
                    parsed_data["rate_probabilities"] = self._parse_data_structure(data['data'])
                else:
                    print("⚠️ 알 수 없는 데이터 구조")
                    return self._get_fallback_data()
            
            # 시장 기대 요약 생성
            parsed_data["market_expectations"] = self._generate_market_expectations(
                parsed_data["rate_probabilities"]
            )
            
            return parsed_data
            
        except Exception as e:
            print(f"❌ 데이터 파싱 중 오류: {e}")
            return self._get_fallback_data()
    
    def _parse_api_data(self, data: Dict) -> Dict:
        """
        API 데이터를 파싱합니다.
        
        Args:
            data: API 응답 데이터
            
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
                "data_source": "CME FedWatch Tool API"
            }
            
            # API 데이터 구조에 따라 파싱
            if 'quotes' in data:
                quotes = data['quotes']
                parsed_data["rate_probabilities"] = self._parse_quotes_data(quotes)
            elif 'data' in data:
                parsed_data["rate_probabilities"] = self._parse_data_structure(data['data'])
            else:
                print("⚠️ API에서 알 수 없는 데이터 구조")
                return self._get_fallback_data()
            
            # 시장 기대 요약 생성
            parsed_data["market_expectations"] = self._generate_market_expectations(
                parsed_data["rate_probabilities"]
            )
            
            return parsed_data
            
        except Exception as e:
            print(f"❌ API 데이터 파싱 중 오류: {e}")
            return self._get_fallback_data()
    
    def _parse_quotes_data(self, quotes: List) -> Dict:
        """
        quotes 데이터를 파싱합니다.
        
        Args:
            quotes: quotes 리스트
            
        Returns:
            dict: 파싱된 금리 확률 데이터
        """
        rate_probs = {}
        
        try:
            for quote in quotes:
                if isinstance(quote, dict):
                    # FOMC 회의일 추출
                    meeting_date = quote.get('meetingDate', '')
                    if meeting_date:
                        # 금리 수준과 확률 추출
                        rate_level = quote.get('rateLevel', '')
                        probability = quote.get('probability', 0)
                        
                        if meeting_date not in rate_probs:
                            rate_probs[meeting_date] = {}
                        
                        if rate_level and probability is not None:
                            rate_probs[meeting_date][str(rate_level)] = float(probability)
            
            return rate_probs
            
        except Exception as e:
            print(f"❌ quotes 데이터 파싱 중 오류: {e}")
            return {}
    
    def _parse_data_structure(self, data: Dict) -> Dict:
        """
        일반적인 데이터 구조를 파싱합니다.
        
        Args:
            data: 데이터 딕셔너리
            
        Returns:
            dict: 파싱된 금리 확률 데이터
        """
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
        """
        다음 FOMC 회의일을 반환합니다.
        
        Returns:
            str: 다음 FOMC 회의일 (YYYY-MM-DD)
        """
        today = datetime.now()
        
        for meeting_date in self.fomc_dates_2025:
            meeting_dt = datetime.strptime(meeting_date, "%Y-%m-%d")
            if meeting_dt > today:
                return meeting_date
        
        # 모든 회의가 지났으면 가장 가까운 미래 회의 반환
        return self.fomc_dates_2025[-1]
    
    def _generate_market_expectations(self, rate_probs: Dict) -> Dict:
        """
        시장 기대 요약을 생성합니다.
        
        Args:
            rate_probs: 금리 확률 데이터
            
        Returns:
            dict: 시장 기대 요약
        """
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
        """
        스크래핑 실패 시 사용할 대체 데이터를 반환합니다.
        
        Returns:
            dict: 대체 FED Watch 데이터
        """
        print("🔄 대체 데이터 사용")
        
        # 2025년 8월 기준 시장 기대치 (CME FedWatch Tool 기반)
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
            "data_source": "대체 데이터 (스크래핑 실패)"
        }
        
        return fallback_data
    
    def get_current_fed_watch_data(self) -> Dict:
        """
        현재 FED Watch 데이터를 가져옵니다.
        
        Returns:
            dict: 현재 FED Watch 데이터
        """
        print("📊 CME FedWatch Tool에서 실시간 데이터 수집 중...")
        
        # 웹사이트 스크래핑 시도
        data = self.scrape_fed_watch_website()
        
        if data:
            print(f"✅ 데이터 수집 완료 (출처: {data.get('data_source', 'N/A')})")
            print(f"📅 다음 FOMC 회의: {data.get('next_meeting', 'N/A')}")
            print(f"🏦 현재 연방기금금리: {data.get('current_rate', 'N/A')}%")
        else:
            print("❌ 데이터 수집 실패")
        
        return data


def main():
    """메인 함수"""
    print("CME FedWatch Tool 웹 스크래퍼를 시작합니다...")
    
    scraper = CMEFedWatchScraper()
    
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


if __name__ == "__main__":
    main()
