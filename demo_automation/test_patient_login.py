import requests
import re

s = requests.Session()
r = s.get('http://127.0.0.1:5000/patient-login')
token = re.search(r'name="csrf_token" value="([^"]+)"', r.text)
token_val = token.group(1) if token else ''
data = {'email': 'vincegonato48@gmail.com', 'password': '12345678', 'csrf_token': token_val}
r2 = s.post('http://127.0.0.1:5000/patient-login', data=data)
print('Status:', r2.status_code, 'Final URL:', r2.url)
print('Flashes:', re.findall(r'flash-[a-z]+">([^<]+)', r2.text))
