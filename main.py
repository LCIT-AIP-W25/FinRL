import requests
import time
import random

# Right now we are storing the data in a dictionary
users = {}

# User class
class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.balance = 10000  # Starting balance for the user
        self.portfolio = {}  # User's stocks and their quantities

    def buy_stock(self, stock_symbol, quantity, price):
        total_cost = price * quantity
        if total_cost > self.balance:
            print(f"Insufficient funds to buy {quantity} shares of {stock_symbol}.")
            return
        
        self.balance -= total_cost
        if stock_symbol in self.portfolio:
            self.portfolio[stock_symbol] += quantity
        else:
            self.portfolio[stock_symbol] = quantity
        print(f"Successfully bought {quantity} shares of {stock_symbol} at ${price:.2f}. Remaining balance: ${self.balance:.2f}")

    def view_portfolio(self):
        if not self.portfolio:
            print("Your portfolio is empty.")
        else:
            for symbol, quantity in self.portfolio.items():
                print(f"{symbol}: {quantity} shares")

# Functions to handle user interaction
def signup():
    username = input("Enter a username: ")
    if username in users:
        print("Username already exists. Please choose a different one.")
        return
    password = input("Enter a password: ")
    users[username] = User(username, password)
    print(f"User {username} created successfully.")

def login():
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    if username in users and users[username].password == password:
        print(f"Welcome back, {username}!")
        return users[username]
    else:
        print("Invalid username or password.")
        return None

def get_latest_stock_price(stock_symbol):
    base_url = "http://127.0.0.1:5000/api/stock"  # Replace with your API's actual URL
    try:
        response = requests.get(base_url, params={"symbol": stock_symbol})
        if response.status_code == 200 and response.json().get("success"):
            return response.json().get("price")
        else:
            print(f"Error fetching stock data for {stock_symbol}: {response.json().get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error connecting to stock API: {e}")
        return None

def buy_stock(user):
    print("Available stocks: AAPL, GOOGL, AMZN, TSLA")
    
    stock_symbol = input("Enter the stock symbol you want to buy: ").upper()
    quantity = int(input("Enter the quantity of stock you want to buy: "))
    price = get_latest_stock_price(stock_symbol)
    
    if price:
        user.buy_stock(stock_symbol, quantity, price)
    else:
        print(f"Could not retrieve the price for {stock_symbol}.")

def view_portfolio(user):
    print("\nYour portfolio:")
    user.view_portfolio()

def main():
    print("Welcome to the Stock Market Simulator!")
    while True:
        action = input("Do you want to sign up, login, or quit? (signup/login/quit): ").lower()
        
        if action == 'signup':
            signup()
        elif action == 'login':
            user = login()
            if user:
                while True:
                    print("\n1. Buy Stock\n2. View Portfolio\n3. Logout")
                    choice = input("Choose an option (1/2/3): ")
                    
                    if choice == '1':
                        buy_stock(user)
                    elif choice == '2':
                        view_portfolio(user)
                    elif choice == '3':
                        print("Logging out...")
                        break
                    else:
                        print("Invalid choice. Please try again.")
        elif action == 'quit':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
