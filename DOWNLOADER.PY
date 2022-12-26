import requests
import requests

url = 'http://schizconnect.org/api/auth/sign_in'
headers = {'Content-Type': 'application/json'}
data = {'email': 'nsh531@gmail.com', 'password': '318962420'}

response = requests.post(url, headers=headers, json=data)

# Check the status code of the response
print(response.status_code)

# Print the headers of the response
print(dict(response.headers))
import requests

url ='http://schizconnect.org/downloadable_files/16028'
headers = {
    'access-token': dict(response.headers)["access-token"],
    'client': dict(response.headers)["client"],
    'uid': dict(response.headers)["uid"]
}

response1 = requests.get(url, headers=headers)

# Check the status code of the response
print(response1.status_code)

# Print the headers of the response
print(response1.headers)

# Save the content of the response (the file) to a local file
with open('16028.7z', 'wb') as f:
    f.write(response1.content)

# Print the content of the response (the JSON data)
print(response1.json())

url ='http://schizconnect.org/downloadable_files/16030'
headers = {
    'access-token': dict(response.headers)["access-token"],
    'client': dict(response.headers)["client"],
    'uid': dict(response.headers)["uid"]
}

response2 = requests.get(url, headers=headers)

# Check the status code of the response
print(response2.status_code)

# Print the headers of the response
print(response2.headers)

# Save the content of the response (the file) to a local file
with open('16030.7z', 'wb') as f:
    f.write(response2.content)

# Print the content of the response (the JSON data)
print(response2.json())
