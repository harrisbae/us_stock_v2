#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
직접 웹 스크래핑 테스트
"""

import sys
import os
sys.path.append('src')

from kostalany_data_loader import safe_request_with_delay, get_enhanced_headers
from bs4 import BeautifulSoup
import time

def test_direct_scraping():
    """직접 웹 스크래핑 테스트"""
    print("🔍 직접 웹 스크래핑 테스트")
    print("=" * 50)
    
    # 테스트할 사이트들
    test_sites = [
        {
            "name": "미국 GDP",
            "url": "https://tradingeconomics.com/united-states/gdp-growth-annual",
            "selector": "span.act-value"
        },
        {
            "name": "미국 인플레이션",
            "url": "https://tradingeconomics.com/united-states/inflation-cpi",
            "selector": "span.act-value"
        },
        {
            "name": "미국 금리",
            "url": "https://tradingeconomics.com/united-states/interest-rate",
            "selector": "span.act-value"
        },
        {
            "name": "미국 실업률",
            "url": "https://tradingeconomics.com/united-states/unemployment-rate",
            "selector": "span.act-value"
        }
    ]
    
    results = []
    
    for i, site in enumerate(test_sites, 1):
        print(f"\n{i}. {site['name']}")
        print(f"   URL: {site['url']}")
        
        try:
            # 3초 지연으로 요청
            print(f"   🔍 데이터 수집 시도...")
            response = safe_request_with_delay(site['url'], delay=3.0, max_retries=2)
            
            if response and response.status_code == 200:
                print(f"   ✅ 응답 성공: 상태 코드 {response.status_code}")
                
                # HTML 파싱
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 데이터 추출
                value_element = soup.select_one(site['selector'])
                if value_element:
                    value = value_element.text.strip()
                    print(f"   📊 추출된 값: {value}")
                    
                    # 숫자로 변환 시도
                    try:
                        numeric_value = float(value)
                        print(f"   🔢 숫자 변환 성공: {numeric_value}")
                        results.append({
                            "site": site['name'],
                            "success": True,
                            "value": value,
                            "numeric_value": numeric_value,
                            "status": response.status_code
                        })
                    except ValueError:
                        print(f"   ⚠️ 숫자 변환 실패: '{value}'는 숫자가 아님")
                        results.append({
                            "site": site['name'],
                            "success": True,
                            "value": value,
                            "numeric_value": None,
                            "status": response.status_code
                        })
                else:
                    print(f"   ❌ 선택자 '{site['selector']}'에서 값 찾을 수 없음")
                    print(f"   🔍 페이지 내용 일부: {response.text[:200]}...")
                    results.append({
                        "site": site['name'],
                        "success": False,
                        "value": None,
                        "numeric_value": None,
                        "status": response.status_code,
                        "error": "선택자에서 값 찾을 수 없음"
                    })
            else:
                print(f"   ❌ 요청 실패")
                results.append({
                    "site": site['name'],
                    "success": False,
                    "value": None,
                    "numeric_value": None,
                    "status": None,
                    "error": "요청 실패"
                })
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            results.append({
                "site": site['name'],
                "success": False,
                "value": None,
                "numeric_value": None,
                "status": None,
                "error": str(e)
            })
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"총 테스트: {total}")
    print(f"성공: {successful}")
    print(f"실패: {total - successful}")
    print(f"성공률: {successful/total*100:.1f}%")
    
    print("\n상세 결과:")
    for result in results:
        status_icon = "✅" if result['success'] else "❌"
        value_info = f"{result['value']}" if result['value'] else "N/A"
        print(f"{status_icon} {result['site']}: {value_info}")
    
    print("\n" + "=" * 50)
    print("✅ 직접 웹 스크래핑 테스트 완료!")

if __name__ == "__main__":
    test_direct_scraping()
