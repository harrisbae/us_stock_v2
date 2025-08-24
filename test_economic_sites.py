#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
경제 데이터 사이트 User-Agent 개선 테스트
"""

import sys
import os
sys.path.append('src')

from kostalany_data_loader import safe_request_with_delay

def test_economic_sites():
    """경제 데이터 사이트 접근 테스트"""
    print("🌐 경제 데이터 사이트 접근 테스트")
    print("=" * 60)
    
    # 테스트할 경제 데이터 사이트들
    test_sites = [
        {
            "name": "Trading Economics - 한국 GDP",
            "url": "https://tradingeconomics.com/south-korea/gdp-growth-annual",
            "expected_status": 200
        },
        {
            "name": "Trading Economics - 미국 GDP", 
            "url": "https://tradingeconomics.com/united-states/gdp-growth-annual",
            "expected_status": 200
        },
        {
            "name": "Focus Economics - 한국 GDP",
            "url": "https://www.focus-economics.com/country-indicator/korea/gdp",
            "expected_status": 200
        },
        {
            "name": "Trading Economics - VIX",
            "url": "https://tradingeconomics.com/vix:ind",
            "expected_status": 200
        },
        {
            "name": "Trading Economics - DXY",
            "url": "https://tradingeconomics.com/dxy:cur",
            "expected_status": 200
        }
    ]
    
    results = []
    
    for i, site in enumerate(test_sites, 1):
        print(f"\n{i}. {site['name']}")
        print(f"   URL: {site['url']}")
        
        try:
            # 3초 지연으로 요청
            response = safe_request_with_delay(site['url'], delay=3.0, max_retries=2)
            
            if response:
                status = response.status_code
                success = status == site['expected_status']
                
                if success:
                    print(f"   ✅ 성공: 상태 코드 {status}")
                    # 간단한 내용 확인
                    content_length = len(response.text)
                    print(f"   📄 응답 크기: {content_length:,} bytes")
                    
                    # 제목이나 키워드 확인
                    if "GDP" in response.text[:1000]:
                        print(f"   🔍 GDP 관련 내용 발견")
                    elif "VIX" in response.text[:1000]:
                        print(f"   🔍 VIX 관련 내용 발견")
                    elif "DXY" in response.text[:1000]:
                        print(f"   🔍 DXY 관련 내용 발견")
                    
                else:
                    print(f"   ⚠️ 예상과 다른 상태 코드: {status} (예상: {site['expected_status']})")
                
                results.append({
                    "site": site['name'],
                    "success": success,
                    "status": status,
                    "content_length": len(response.text) if response else 0
                })
                
            else:
                print(f"   ❌ 요청 실패")
                results.append({
                    "site": site['name'],
                    "success": False,
                    "status": None,
                    "content_length": 0
                })
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            results.append({
                "site": site['name'],
                "success": False,
                "status": None,
                "content_length": 0,
                "error": str(e)
            })
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"총 테스트: {total}")
    print(f"성공: {successful}")
    print(f"실패: {total - successful}")
    print(f"성공률: {successful/total*100:.1f}%")
    
    print("\n상세 결과:")
    for result in results:
        status_icon = "✅" if result['success'] else "❌"
        print(f"{status_icon} {result['site']}: {result['status']} ({result['content_length']:,} bytes)")
    
    print("\n" + "=" * 60)
    print("✅ 경제 데이터 사이트 테스트 완료!")

if __name__ == "__main__":
    test_economic_sites()
