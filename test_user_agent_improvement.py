#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
User-Agent 개선 테스트 스크립트
"""

import sys
import os
sys.path.append('src')

from kostalany_data_loader import get_enhanced_headers, get_random_user_agent, safe_request_with_delay

def test_user_agent_improvement():
    """User-Agent 개선 사항 테스트"""
    print("🔧 User-Agent 개선 테스트")
    print("=" * 50)
    
    # 1. 향상된 헤더 테스트
    print("\n1. 향상된 헤더 테스트:")
    headers = get_enhanced_headers()
    for key, value in headers.items():
        print(f"   {key}: {value}")
    
    # 2. 랜덤 User-Agent 테스트
    print("\n2. 랜덤 User-Agent 테스트:")
    for i in range(3):
        ua = get_random_user_agent()
        print(f"   시도 {i+1}: {ua}")
    
    # 3. 간단한 웹사이트 접근 테스트
    print("\n3. 웹사이트 접근 테스트:")
    test_urls = [
        "https://httpbin.org/user-agent",
        "https://httpbin.org/headers"
    ]
    
    for url in test_urls:
        print(f"\n   테스트 URL: {url}")
        response = safe_request_with_delay(url, delay=1.0)
        if response:
            print(f"   ✅ 성공: 상태 코드 {response.status_code}")
            if "user-agent" in url:
                # User-Agent 확인
                data = response.json()
                print(f"   📱 User-Agent: {data.get('user-agent', 'N/A')}")
        else:
            print(f"   ❌ 실패")
    
    print("\n" + "=" * 50)
    print("✅ User-Agent 개선 테스트 완료!")

if __name__ == "__main__":
    test_user_agent_improvement()
