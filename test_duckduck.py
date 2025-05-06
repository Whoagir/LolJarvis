from duckai import DuckAI

def send_request(query, model="gpt-4o-mini"):
    duckai = DuckAI()
    response = duckai.chat(query, model=model)
    print(response)

if __name__ == "__main__":
    query = ("Ответьте мне, что такое инкапсуляция в питоне в 3 предложениях")
    send_request(query)