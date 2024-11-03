import africastalking
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Africa's Talking API credentials from .env
username = os.getenv('AT_USER_NAME')
api_key = os.getenv('AT_API_KEY')

print("Username:", username)
print("API Key:", api_key)

# Initialize Africa's Talking with loaded credentials
try:
    africastalking.initialize(username, api_key)
    print("Initialization successful!")
except RuntimeError as e:
    print("Initialization error:", e)
