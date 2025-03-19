# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain",
#     "langchain-openai",
#     "python-dotenv",
# ]
# ///

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Check and set required environment variables
required_vars = {
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
    "LANGSMITH_API_KEY": "<redacted>",
    "LANGSMITH_PROJECT": "w267-final-project-rag-pipeline",
    "OPENAI_API_KEY": "<redacted>",
}

for var, default_value in required_vars.items():
    if not os.getenv(var):
        os.environ[var] = default_value
        print(f"Setting {var}={default_value}")

# Verify all required variables are set
missing_vars = [var for var, _ in required_vars.items() if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Run the LangChain code with LangSmith tracing enabled
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
result = llm.invoke("Tell me about LangSmith in one sentence.")
print(f"Result: {result.content}")
print(
    "Check your LangSmith dashboard for the trace in project: "
    + os.getenv("LANGSMITH_PROJECT")
)
