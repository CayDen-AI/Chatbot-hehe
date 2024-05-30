from flask import Flask, render_template, request
from query import read_db, create_prompt, create_chain
from base_model import get_model
from utils import extract_answer
import requests


app = Flask(__name__)

FAISS_PATH = 'faiss'
MODEL_NAME = 'vilm/vinallama-2.7b-chat'
db = read_db(FAISS_PATH)
llm = get_model(MODEL_NAME)
prompt = create_prompt()
llm_chain = create_chain(prompt, llm, db)


@app.route('/', methods=['GET', 'POST'])
def query():
    bot_chat = ''
    if request.method == 'POST':
        user_chat = request.form['user_chat']
        if user_chat:
            response = llm_chain.invoke({'query': user_chat})
            bot_chat = extract_answer(response['result'])
    return render_template('index.html', bot_chat=bot_chat)


if __name__ == '__main__':
    app.run()
