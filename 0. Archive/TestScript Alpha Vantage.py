import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#   url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=URCCM7R572UT24TU'

api_key = 'URCCM7R572UT24TU'
ticker = 'IBM'
function = 'TIME_SERIES_DAILY_ADJUSTED'
output_size = 'compact' #full otherwise

url =f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&outputsize={output_size}&apikey={api_key}'

r = requests.get(url)
data = r.json()

print(data)