import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_api():
    """Test the OpenAI API connection with a simple prompt."""
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables")
            return False

        # Get model from environment and debug print
        print("\nEnvironment variables:")
        print(f"OPENAI_MODEL = {os.getenv('OPENAI_MODEL')}")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using model: {model}")

        # Initialize client
        client = OpenAI(api_key=api_key)

        # Make a simple test call
        print("\nMaking test API call...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'Hello, API test successful!'"}],
            temperature=0.3,
            max_tokens=50
        )

        # Print response
        print("\nResponse received:")
        print(response.choices[0].message.content)
        
        # Print token usage
        print("\nToken usage:")
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"\nError testing OpenAI API: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing OpenAI API connection...")
    success = test_openai_api()
    if success:
        print("\n✅ API test completed successfully!")
    else:
        print("\n❌ API test failed!") 