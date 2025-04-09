#!/usr/bin/env python3
from langsmith import Client
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize the LangSmith client
client = Client()

# Get all experiments from your project
# The project name should match your LANGCHAIN_PROJECT env variable
experiments = client.list_runs()

# Extract experiment names/ids
experiment_names = [{"run_id": run.id, "name": run.name} for run in experiments]

# Write to file
with open("langsmith_experiments.json", "w") as f:
    json.dump(experiment_names, f, indent=2)

print(f"Saved {len(experiment_names)} experiments to langsmith_experiments.json")