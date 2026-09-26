import os
os.environ["XXHASH_FORCE_PURE_PYTHON"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from agents import create_deep_search_supervisor


def run_deep_search(topic: str):
    supervisor = create_deep_search_supervisor()
    
    print(f"🚀 Starting Autonomous Deep Search for: '{topic}'\n")
    
    response = supervisor.invoke({
        "messages": [
            {"role": "user", "content": f"Conduct a deep research report on: {topic}"}
        ]
    })

    # The final message from the supervisor contains the refined report
    final_report = response["messages"][-1].content
    print("\n================ FINAL RESEARCH REPORT ================\n")
    print(final_report)

if __name__ == "__main__":
    user_query = input("Enter deep search topic: ")
    run_deep_search(user_query.strip())