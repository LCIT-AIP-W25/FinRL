import requests

def test_stock_api():
    # Base URL for the API
    base_url = "http://127.0.0.1:5000/api/stock"

    print("Starting API tests...")

    # Test case 1: Valid stock symbol
    symbol = "AAPL"
    response = requests.get(base_url, params={"symbol": symbol})
    if response.status_code == 200 and response.json().get("success"):
        print(f"Test 1 Passed: Valid symbol '{symbol}' returned valid data.")
    else:
        print(f"Test 1 Failed: Valid symbol '{symbol}' did not return valid data.")
        print("Response:", response.json())

    # Test case 2: Missing stock symbol
    response = requests.get(base_url)
    if response.status_code == 400 and "Missing 'symbol' parameter" in response.json().get("message", ""):
        print("Test 2 Passed: Missing symbol returned appropriate error.")
    else:
        print("Test 2 Failed: Missing symbol did not return appropriate error.")
        print("Response:", response.json())

    # Test case 3: Invalid stock symbol
    invalid_symbol = "INVALID"
    response = requests.get(base_url, params={"symbol": invalid_symbol})
    if response.status_code == 404 and f"Stock symbol '{invalid_symbol}' not found" in response.json().get("message", ""):
        print(f"Test 3 Passed: Invalid symbol '{invalid_symbol}' returned appropriate error.")
    else:
        print(f"Test 3 Failed: Invalid symbol '{invalid_symbol}' did not return appropriate error.")
        print("Response:", response.json())

    print("API tests completed.")

# Run the test function
if __name__ == "__main__":
    test_stock_api()
