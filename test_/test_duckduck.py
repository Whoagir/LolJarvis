from duckai import DuckAI


def send_request(query, model="gpt-4o-mini"):
    duckai = DuckAI()
    response = duckai.chat(query, model=model)
    print(response)


if __name__ == "__main__":
    p = "Представь ты на собеседование мидл python разработчик, тебе надо коротко ответить на вопрос. Учитывай, что я могу опечататься в предложение, но тебе надо стараться понять о чем разговор и дать ответ. Вот вопрос:"
    s = p + " Знаешь чем отличаются потоки и процессы "
    query = (s)
    send_request(query)
