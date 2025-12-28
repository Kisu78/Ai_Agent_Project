from google import genai
from google.genai import types
import requests
from datetime import datetime

# -------------------------
# Configure Gemini Client
# -------------------------
client = genai.Client(
    #Here i provide my Api key But i Will Exhuast Shortly 
    api_key="AIzaSyAEiGuCRT3rDhZB6-E6zwgbfF3koKMSlq0"  # Replace with your Gemini API key
    
    # api_key="YOUR_GEMINI_API_KEY"  # Replace with your Gemini API key
)

# -------------------------
# TOOLS
# -------------------------
def crypto_currency(coin: str):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "inr", "ids": coin}
    return requests.get(url, params=params).json()


def weather_information(city: str):
    url = "http://api.weatherapi.com/v1/current.json"

    # Here i Provide my Weather key if it exhuast Use Your Weather key
    params = {"key": "5477b42907744bfaaec44619252812", "q": city, "aqi": "no"}
    return requests.get(url, params=params).json()

def current_datetime():
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "Local System Time"
    }

# -------------------------
# TOOL SCHEMAS
# -------------------------
crypto_tool = types.FunctionDeclaration(
    name="crypto_currency",
    description="Get current cryptocurrency price and related information",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"coin": types.Schema(type=types.Type.STRING, description="Cryptocurrency name")},
        required=["coin"]
    )
)

weather_tool = types.FunctionDeclaration(
    name="weather_information",
    description="Get current weather information of a city",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"city": types.Schema(type=types.Type.STRING, description="City name")},
        required=["city"]
    )
)

datetime_tool = types.FunctionDeclaration(
    name="current_datetime",
    description="Get current real-time date and time",
    parameters=types.Schema(type=types.Type.OBJECT, properties={})
)

tools = types.Tool(function_declarations=[crypto_tool, weather_tool, datetime_tool])
tool_functions = {
    "crypto_currency": crypto_currency,
    "weather_information": weather_information,
    "current_datetime": current_datetime
}

# -------------------------
# SYSTEM INSTRUCTION (sent as USER role)
# -------------------------
system_instruction = """
You are an AI Agent.
Answer questions directly.
If Question Dont Require Any Tool So Give Direct Answers
If user asks:
- cryptocurrency → call crypto_currency
- weather → call weather_information
- date/time → call current_datetime
Always provide accurate real-time info using tools when required.
"""

# -------------------------
# MEMORY
# -------------------------
history = [
    types.Content(
        role="user",  # Use "user" instead of "system"
        parts=[types.Part(text=system_instruction)]
    )
]

# -------------------------
# AGENT LOOP
# -------------------------
def run_agent():
    while True:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(tools=[tools])
        )

        # TOOL CALL
        if response.function_calls:
            function_call = response.function_calls[0]
            name = function_call.name
            args = function_call.args
            print("🔧 Tool called:", name)

            tool_response = tool_functions[name](**args)

            # Append function call
            history.append(
                types.Content(
                    role="assistant",  # AI role
                    parts=[types.Part(function_call=function_call)]
                )
            )

            # Append function response
            history.append(
                types.Content(
                    role="user",  # back to user role
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=name,
                        response={"result": tool_response}
                    ))]
                )
            )
        else:
            print("\n🤖 Gemini:", response.text)
            history.append(
                types.Content(
                    role="assistant",
                    parts=[types.Part(text=response.text)]
                )
            )
            break

# -------------------------
# USER LOOP
# -------------------------
while True:
    question = input("\nAsk me anything (Press exit For Leave): ")
    if question.lower() == "exit":
        break
    history.append(
        types.Content(
            role="user",
            parts=[types.Part(text=question)]
        )
    )
    run_agent()
