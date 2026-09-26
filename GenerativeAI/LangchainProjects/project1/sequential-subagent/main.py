import json
from rich import print
from langchain_core.messages import HumanMessage
from agents import search_agent,web_scrapping_agent,deep_search_summarizer,critic_summarizer

def run_deep_search(topic:str)->dict:

    state = dict()

    # 1. Search web:
    search_response = search_agent().invoke({
        "messages":[{"role":"user","content":f"Find recent and detailed information about user query = {topic}"}]
    })

    # print(search_response)

    web_result = search_response.get("messages")[-1].content
    state["web_search_result"] = web_result
    
    print("\nWeb Search Result:\n",web_result[:10000])


    # 2. Web scrapping:

    scrapping_response = web_scrapping_agent().invoke({
        "messages":[{"role":"user","content":f"Here are the web result content = {web_result[:1000]}"}]
    })

    scarping_result = scrapping_response.get("messages")[-1].content
    state["scrapping_result"] = scarping_result
    
    print("\nScrapping Result:\n",scarping_result[:1000])


    # 3. deep search summary:

    full_result = f"""
    web-search-result = {state['web_search_result']}

    web-scrapping-result = {state['scrapping_result']}
    """

    deep_search_result = deep_search_summarizer.invoke({
        "topic":topic,
        "research":full_result
    })

    print("\ndeep_search_result:\n",deep_search_result[:1000])

    state["deep_search_result"] = deep_search_result


    # 4. Critic summary:

    critic_result = critic_summarizer.invoke({
        "report":state["deep_search_result"]
    })

    print("\ncritic_result",critic_result[:1000])

    state["critic_result"] = critic_result

    return state






if __name__ == "__main__":

    user_query = input("Enter deep searh topic: ")
    
    run_deep_search(user_query.strip())