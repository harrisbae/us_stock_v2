import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PERPLEXITY_API_KEY")
url = "https://api.perplexity.ai/chat/completions"
headers = {
"Authorization": f"Bearer {api_key}",
"Content-Type": "application/json"
}
data = {
"model": "sonar-pro",
"messages": [
   {"role": "user", "content": "AAPL 관련 최신 뉴스 1개만 한글로 요약해줘."}
]
}
response = requests.post(url, headers=headers, json=data)
print("HTTP 상태 코드:", response.status_code)
print("응답 본문:", response.text)
