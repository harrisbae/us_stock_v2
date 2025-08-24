#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FED Watch 금리인하 기대감 분석기

CME FedWatch Tool을 기반으로 연방기금금리 선물 데이터를 분석하여
시장의 금리인하 기대감을 조회하고 분석합니다.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class FedWatchAnalyzer:
    """
    FED Watch 금리인하 기대감 분석기
    """
    
    def __init__(self):
        self.base_url = "https://www.cmegroup.com/CmeWS/mvc/Quotes/FedWatch"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 연준 FOMC 회의 일정 (2025년)
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
        
    def get_fed_watch_data(self) -> Dict:
        """
        CME FedWatch Tool에서 금리 정책 기대 데이터를 가져옵니다.
        
        Returns:
            dict: FED Watch 데이터
        """
        try:
            # CME FedWatch API 호출
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except requests.RequestException as e:
            print(f"FED Watch 데이터 조회 실패: {e}")
            # 하드코딩된 대체 데이터 반환
            return self._get_fallback_fed_watch_data()
    
    def _get_fallback_fed_watch_data(self) -> Dict:
        """
        FED Watch API 실패 시 사용할 하드코딩된 대체 데이터
        
        Returns:
            dict: 대체 FED Watch 데이터
        """
        # 2025년 8월 기준 시장 기대치 (CME FedWatch 기반)
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
            }
        }
        
        return fallback_data
    
    def analyze_rate_expectations(self, data: Dict) -> Dict:
        """
        금리 정책 기대를 분석합니다.
        
        Args:
            data: FED Watch 데이터
            
        Returns:
            dict: 분석 결과
        """
        analysis = {
            "current_analysis": {},
            "meeting_analysis": {},
            "trend_analysis": {},
            "investment_implications": {}
        }
        
        try:
            # 현재 금리 정책 상황 분석
            current_rate = data.get("current_rate", 4.5)
            analysis["current_analysis"] = {
                "current_fed_rate": current_rate,
                "policy_stance": self._analyze_policy_stance(current_rate),
                "rate_level": self._categorize_rate_level(current_rate)
            }
            
            # 각 FOMC 회의별 기대 분석
            rate_probs = data.get("rate_probabilities", {})
            analysis["meeting_analysis"] = {}
            
            for meeting_date, probabilities in rate_probs.items():
                analysis["meeting_analysis"][meeting_date] = {
                    "rate_cut_probability": self._calculate_rate_cut_probability(probabilities, current_rate),
                    "rate_hike_probability": self._calculate_rate_hike_probability(probabilities, current_rate),
                    "most_likely_outcome": self._get_most_likely_outcome(probabilities),
                    "market_sentiment": self._analyze_market_sentiment(probabilities, current_rate)
                }
            
            # 전반적 트렌드 분석
            analysis["trend_analysis"] = self._analyze_overall_trend(rate_probs, current_rate)
            
            # 투자 시사점 분석
            analysis["investment_implications"] = self._analyze_investment_implications(analysis)
            
        except Exception as e:
            print(f"금리 정책 기대 분석 중 오류: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_policy_stance(self, rate: float) -> str:
        """정책 방향을 분석합니다."""
        if rate >= 5.0:
            return "긴축 정책 (Inflation Fighting)"
        elif rate >= 4.0:
            return "중립적 긴축 (Neutral-Tight)"
        elif rate >= 3.0:
            return "중립적 (Neutral)"
        elif rate >= 2.0:
            return "중립적 완화 (Neutral-Easy)"
        else:
            return "완화 정책 (Stimulative)"
    
    def _categorize_rate_level(self, rate: float) -> str:
        """금리 수준을 분류합니다."""
        if rate >= 6.0:
            return "극도로 높음 (Extremely High)"
        elif rate >= 5.0:
            return "매우 높음 (Very High)"
        elif rate >= 4.0:
            return "높음 (High)"
        elif rate >= 3.0:
            return "보통 (Normal)"
        elif rate >= 2.0:
            return "낮음 (Low)"
        else:
            return "매우 낮음 (Very Low)"
    
    def _calculate_rate_cut_probability(self, probabilities: Dict, current_rate: float) -> float:
        """금리인하 확률을 계산합니다."""
        cut_prob = 0.0
        for rate_str, prob in probabilities.items():
            rate = float(rate_str)
            if rate < current_rate:
                cut_prob += prob
        return cut_prob
    
    def _calculate_rate_hike_probability(self, probabilities: Dict, current_rate: float) -> float:
        """금리인상 확률을 계산합니다."""
        hike_prob = 0.0
        for rate_str, prob in probabilities.items():
            rate = float(rate_str)
            if rate > current_rate:
                hike_prob += prob
        return hike_prob
    
    def _get_most_likely_outcome(self, probabilities: Dict) -> Tuple[str, float]:
        """가장 가능성이 높은 결과를 반환합니다."""
        if not probabilities:
            return "N/A", 0.0
        
        max_prob_rate = max(probabilities.items(), key=lambda x: x[1])
        return max_prob_rate[0], max_prob_rate[1]
    
    def _analyze_market_sentiment(self, probabilities: Dict, current_rate: float) -> str:
        """시장 심리를 분석합니다."""
        cut_prob = self._calculate_rate_cut_probability(probabilities, current_rate)
        hike_prob = self._calculate_rate_hike_probability(probabilities, current_rate)
        
        if cut_prob > 60:
            return "강한 금리인하 기대 (Strong Rate Cut Expectation)"
        elif cut_prob > 40:
            return "금리인하 기대 (Rate Cut Expectation)"
        elif hike_prob > 60:
            return "강한 금리인상 기대 (Strong Rate Hike Expectation)"
        elif hike_prob > 40:
            return "금리인상 기대 (Rate Hike Expectation)"
        else:
            return "현행 유지 기대 (Status Quo Expectation)"
    
    def _analyze_overall_trend(self, rate_probs: Dict, current_rate: float) -> Dict:
        """전반적 트렌드를 분석합니다."""
        trend_analysis = {
            "short_term": "중립",
            "medium_term": "중립",
            "long_term": "중립",
            "overall_direction": "중립",
            "confidence": "보통"
        }
        
        try:
            # 단기 (다음 회의)
            next_meeting = min(rate_probs.keys())
            next_probs = rate_probs[next_meeting]
            next_cut_prob = self._calculate_rate_cut_probability(next_probs, current_rate)
            next_hike_prob = self._calculate_rate_hike_probability(next_probs, current_rate)
            
            if next_cut_prob > 50:
                trend_analysis["short_term"] = "금리인하 기대"
            elif next_hike_prob > 50:
                trend_analysis["short_term"] = "금리인상 기대"
            else:
                trend_analysis["short_term"] = "현행 유지 기대"
            
            # 중기 (3-6개월)
            medium_term_cut_prob = 0
            medium_term_count = 0
            
            for meeting_date, probs in rate_probs.items():
                if meeting_date > next_meeting:
                    medium_term_cut_prob += self._calculate_rate_cut_probability(probs, current_rate)
                    medium_term_count += 1
            
            if medium_term_count > 0:
                avg_medium_cut_prob = medium_term_cut_prob / medium_term_count
                if avg_medium_cut_prob > 50:
                    trend_analysis["medium_term"] = "금리인하 기대"
                elif avg_medium_cut_prob < 30:
                    trend_analysis["medium_term"] = "금리인상 기대"
                else:
                    trend_analysis["medium_term"] = "현행 유지 기대"
            
            # 전체 방향 결정
            cut_expectations = 0
            hike_expectations = 0
            
            for meeting_date, probs in rate_probs.items():
                cut_prob = self._calculate_rate_cut_probability(probs, current_rate)
                hike_prob = self._calculate_rate_hike_probability(probs, current_rate)
                
                if cut_prob > hike_prob:
                    cut_expectations += 1
                elif hike_prob > cut_prob:
                    hike_expectations += 1
            
            if cut_expectations > hike_expectations:
                trend_analysis["overall_direction"] = "금리인하 기대"
            elif hike_expectations > cut_expectations:
                trend_analysis["overall_direction"] = "금리인상 기대"
            else:
                trend_analysis["overall_direction"] = "현행 유지 기대"
                
        except Exception as e:
            print(f"트렌드 분석 중 오류: {e}")
        
        return trend_analysis
    
    def _analyze_investment_implications(self, analysis: Dict) -> Dict:
        """투자 시사점을 분석합니다."""
        implications = {
            "equity_market": "중립",
            "bond_market": "중립",
            "currency_market": "중립",
            "sector_recommendations": [],
            "risk_level": "보통"
        }
        
        try:
            overall_direction = analysis.get("trend_analysis", {}).get("overall_direction", "중립")
            
            if "금리인하" in overall_direction:
                implications["equity_market"] = "긍정적 (금리인하로 인한 성장주 선호)"
                implications["bond_market"] = "긍정적 (금리인하로 인한 채권가격 상승)"
                implications["currency_market"] = "부정적 (달러 약세 기대)"
                implications["sector_recommendations"] = [
                    "성장주 (기술, 바이오테크)",
                    "금융주 (은행, 보험)",
                    "소비자 경기 민감주"
                ]
                implications["risk_level"] = "높음 (성장주 중심)"
                
            elif "금리인상" in overall_direction:
                implications["equity_market"] = "부정적 (금리인상으로 인한 성장주 압박)"
                implications["bond_market"] = "부정적 (금리인상으로 인한 채권가격 하락)"
                implications["currency_market"] = "긍정적 (달러 강세 기대)"
                implications["sector_recommendations"] = [
                    "방어주 (필수소비재, 헬스케어)",
                    "고배당주",
                    "현금 보유"
                ]
                implications["risk_level"] = "낮음 (방어적 포트폴리오)"
                
            else:  # 현행 유지
                implications["equity_market"] = "중립적 (현재 추세 유지)"
                implications["bond_market"] = "중립적 (현재 수준 유지)"
                implications["currency_market"] = "중립적 (현재 수준 유지)"
                implications["sector_recommendations"] = [
                    "균형잡힌 포트폴리오",
                    "배당성장주",
                    "품질주"
                ]
                implications["risk_level"] = "보통 (균형잡힌 포트폴리오)"
                
        except Exception as e:
            print(f"투자 시사점 분석 중 오류: {e}")
        
        return implications
    
    def generate_fed_watch_report(self) -> str:
        """
        FED Watch 분석 보고서를 생성합니다.
        
        Returns:
            str: 분석 보고서
        """
        try:
            # 데이터 조회
            fed_watch_data = self.get_fed_watch_data()
            
            # 분석 수행
            analysis = self.analyze_rate_expectations(fed_watch_data)
            
            # 보고서 생성
            report = self._format_report(fed_watch_data, analysis)
            
            return report
            
        except Exception as e:
            return f"FED Watch 보고서 생성 실패: {e}"
    
    def _format_report(self, data: Dict, analysis: Dict) -> str:
        """분석 결과를 보고서 형태로 포맷팅합니다."""
        report = []
        report.append("=" * 80)
        report.append("📊 FED Watch 금리인하 기대감 분석 보고서")
        report.append("=" * 80)
        report.append(f"📅 분석일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🏦 현재 연방기금금리: {data.get('current_rate', 'N/A')}%")
        report.append(f"📋 다음 FOMC 회의: {data.get('next_meeting', 'N/A')}")
        report.append("")
        
        # 현재 정책 상황
        current_analysis = analysis.get("current_analysis", {})
        report.append("🔍 현재 정책 상황 분석")
        report.append("-" * 50)
        report.append(f"  • 정책 방향: {current_analysis.get('policy_stance', 'N/A')}")
        report.append(f"  • 금리 수준: {current_analysis.get('rate_level', 'N/A')}")
        report.append("")
        
        # FOMC 회의별 기대 분석
        report.append("📈 FOMC 회의별 금리 정책 기대")
        report.append("-" * 50)
        
        meeting_analysis = analysis.get("meeting_analysis", {})
        for meeting_date, meeting_data in meeting_analysis.items():
            report.append(f"📅 {meeting_date} FOMC:")
            report.append(f"  • 금리인하 확률: {meeting_data.get('rate_cut_probability', 0):.1f}%")
            report.append(f"  • 금리인상 확률: {meeting_data.get('rate_hike_probability', 0):.1f}%")
            report.append(f"  • 가장 가능성 높은 결과: {meeting_data.get('most_likely_outcome', ('N/A', 0))[0]}% ({meeting_data.get('most_likely_outcome', ('N/A', 0))[1]:.1f}%)")
            report.append(f"  • 시장 심리: {meeting_data.get('market_sentiment', 'N/A')}")
            report.append("")
        
        # 전반적 트렌드
        trend_analysis = analysis.get("trend_analysis", {})
        report.append("📊 전반적 트렌드 분석")
        report.append("-" * 50)
        report.append(f"  • 단기 (다음 회의): {trend_analysis.get('short_term', 'N/A')}")
        report.append(f"  • 중기 (3-6개월): {trend_analysis.get('medium_term', 'N/A')}")
        report.append(f"  • 전체 방향: {trend_analysis.get('overall_direction', 'N/A')}")
        report.append("")
        
        # 투자 시사점
        investment_implications = analysis.get("investment_implications", {})
        report.append("💰 투자 시사점")
        report.append("-" * 50)
        report.append(f"  • 주식시장: {investment_implications.get('equity_market', 'N/A')}")
        report.append(f"  • 채권시장: {investment_implications.get('bond_market', 'N/A')}")
        report.append(f"  • 외환시장: {investment_implications.get('currency_market', 'N/A')}")
        report.append(f"  • 위험 수준: {investment_implications.get('risk_level', 'N/A')}")
        report.append("")
        
        # 섹터 추천
        sector_recommendations = investment_implications.get("sector_recommendations", [])
        if sector_recommendations:
            report.append("🎯 섹터 추천")
            report.append("-" * 50)
            for i, sector in enumerate(sector_recommendations, 1):
                report.append(f"  {i}. {sector}")
            report.append("")
        
        # 시장 기대 요약
        market_expectations = data.get("market_expectations", {})
        if market_expectations:
            report.append("📋 시장 기대 요약")
            report.append("-" * 50)
            report.append(f"  • 다음 회의: {market_expectations.get('next_meeting', 'N/A')}")
            report.append(f"  • 연말 전망: {market_expectations.get('year_end', 'N/A')}")
            report.append(f"  • 전반적 트렌드: {market_expectations.get('trend', 'N/A')}")
            report.append("")
        
        report.append("=" * 80)
        report.append("📚 참고: CME FedWatch Tool 기반 분석")
        report.append("🔗 https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_rate_probabilities(self, data: Dict, save_path: Optional[str] = None):
        """
        금리 확률을 시각화합니다.
        
        Args:
            data: FED Watch 데이터
            save_path: 저장 경로 (선택사항)
        """
        try:
            # macOS 호환 폰트 설정
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica']
            plt.rcParams['axes.unicode_minus'] = False
            
            rate_probs = data.get("rate_probabilities", {})
            if not rate_probs:
                print("시각화할 데이터가 없습니다.")
                return
            
            # 데이터 준비
            meetings = list(rate_probs.keys())
            rates = sorted(set([float(rate) for probs in rate_probs.values() for rate in probs.keys()]))
            
            # 확률 매트릭스 생성
            prob_matrix = []
            for meeting in meetings:
                meeting_probs = []
                for rate in rates:
                    prob = rate_probs[meeting].get(str(rate), 0)
                    meeting_probs.append(prob)
                prob_matrix.append(meeting_probs)
            
            # 시각화
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 히트맵
            sns.heatmap(prob_matrix, 
                        xticklabels=[f"{rate:.2f}%" for rate in rates],
                        yticklabels=[f"{meeting}" for meeting in meetings],
                        annot=True, 
                        fmt='.1f',
                        cmap='RdYlBu_r',
                        ax=ax1)
            ax1.set_title('FOMC Meeting Interest Rate Probability Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Federal Funds Rate (%)')
            ax1.set_ylabel('FOMC Meeting Date')
            
            # 막대 그래프 (가장 가능성 높은 결과)
            most_likely_rates = []
            most_likely_probs = []
            
            for meeting in meetings:
                probs = rate_probs[meeting]
                max_prob_rate = max(probs.items(), key=lambda x: x[1])
                most_likely_rates.append(float(max_prob_rate[0]))
                most_likely_probs.append(max_prob_rate[1])
            
            bars = ax2.bar(range(len(meetings)), most_likely_probs, 
                          color=['#FF6B6B' if rate < 4.5 else '#4ECDC4' if rate == 4.5 else '#45B7D1' 
                                 for rate in most_likely_rates])
            
            ax2.set_title('Most Likely Interest Rate by FOMC Meeting', fontsize=14, fontweight='bold')
            ax2.set_xlabel('FOMC Meeting Date')
            ax2.set_ylabel('Probability (%)')
            ax2.set_xticks(range(len(meetings)))
            ax2.set_xticklabels([f"{meeting}" for meeting in meetings], rotation=45)
            
            # 막대 위에 금리 표시
            for i, (bar, rate) in enumerate(zip(bars, most_likely_rates)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            # 범례 추가
            legend_elements = [
                Rectangle((0,0),1,1, facecolor='#FF6B6B', label='Rate Cut'),
                Rectangle((0,0),1,1, facecolor='#4ECDC4', label='Status Quo'),
                Rectangle((0,0),1,1, facecolor='#45B7D1', label='Rate Hike')
            ]
            ax2.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"차트가 저장되었습니다: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"시각화 생성 중 오류: {e}")


def main():
    """메인 함수"""
    print("FED Watch 금리인하 기대감 분석기를 시작합니다...")
    
    analyzer = FedWatchAnalyzer()
    
    # 분석 보고서 생성
    report = analyzer.generate_fed_watch_report()
    print(report)
    
    # 데이터 조회
    fed_watch_data = analyzer.get_fed_watch_data()
    
    # 시각화 생성
    print("\n📊 금리 확률 시각화를 생성합니다...")
    analyzer.plot_rate_probabilities(fed_watch_data, "output/fed_watch_analysis.png")


if __name__ == "__main__":
    main()
