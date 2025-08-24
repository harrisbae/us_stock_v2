#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTML 구조 디버깅 스크립트
"""

import sys
import os
sys.path.append('src')

from kostalany_data_loader import safe_request_with_delay
from bs4 import BeautifulSoup
import re

def debug_html_structure():
    """HTML 구조 디버깅"""
    print("🔍 HTML 구조 디버깅")
    print("=" * 50)
    
    # 테스트할 사이트
    url = "https://tradingeconomics.com/united-states/gdp-growth-annual"
    print(f"URL: {url}")
    
    try:
        print(f"🔍 데이터 수집 시도...")
        response = safe_request_with_delay(url, delay=2.0, max_retries=1)
        
        if response and response.status_code == 200:
            print(f"✅ 응답 성공: 상태 코드 {response.status_code}")
            
            # HTML 파싱
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 페이지 제목 확인
            title = soup.find('title')
            if title:
                print(f"📄 페이지 제목: {title.text.strip()}")
            
            # 다양한 선택자 시도
            selectors_to_try = [
                "span.act-value",
                ".act-value",
                "span[class*='act']",
                "span[class*='value']",
                "span[class*='current']",
                "span[class*='latest']",
                "td[class*='act']",
                "td[class*='value']",
                "div[class*='act']",
                "div[class*='value']",
                "span",
                "td",
                "div"
            ]
            
            print(f"\n🔍 다양한 선택자로 값 찾기 시도...")
            
            for selector in selectors_to_try:
                try:
                    elements = soup.select(selector)
                    if elements:
                        print(f"\n✅ 선택자 '{selector}'에서 {len(elements)}개 요소 발견:")
                        for i, elem in enumerate(elements[:5]):  # 처음 5개만 표시
                            text = elem.text.strip()
                            if text and len(text) < 100:  # 너무 긴 텍스트 제외
                                print(f"   {i+1}. '{text}' (태그: {elem.name}, 클래스: {elem.get('class', 'N/A')})")
                        
                        # 숫자 값이 있는지 확인
                        numeric_values = []
                        for elem in elements:
                            text = elem.text.strip()
                            if text and re.match(r'^-?\d+\.?\d*%?$', text):
                                numeric_values.append(text)
                        
                        if numeric_values:
                            print(f"   🎯 숫자 값 발견: {numeric_values[:3]}")  # 처음 3개만 표시
                        break
                except Exception as e:
                    print(f"   ❌ 선택자 '{selector}' 오류: {e}")
            
            # 페이지 내용에서 숫자 패턴 찾기
            print(f"\n🔍 페이지 내용에서 숫자 패턴 찾기...")
            text_content = soup.get_text()
            
            # GDP 관련 숫자 패턴 찾기
            gdp_patterns = [
                r'(\d+\.?\d*)%?\s*\(GDP\)',
                r'GDP.*?(\d+\.?\d*)%',
                r'(\d+\.?\d*)%?\s*Growth',
                r'Growth.*?(\d+\.?\d*)%'
            ]
            
            for pattern in gdp_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                if matches:
                    print(f"   🎯 GDP 패턴 '{pattern}'에서 발견: {matches[:3]}")
            
            # 일반적인 퍼센트 값 찾기
            percent_pattern = r'(\d+\.?\d*)%'
            percent_matches = re.findall(percent_pattern, text_content)
            if percent_matches:
                print(f"   📊 퍼센트 값들: {percent_matches[:10]}")  # 처음 10개만 표시
            
            # 페이지 소스 일부 저장
            print(f"\n💾 페이지 소스 일부를 'debug_gdp_page.html'에 저장...")
            with open('debug_gdp_page.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"   ✅ 저장 완료: debug_gdp_page.html")
            
        else:
            print(f"❌ 요청 실패")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    print("\n" + "=" * 50)
    print("✅ HTML 구조 디버깅 완료!")

if __name__ == "__main__":
    debug_html_structure()
