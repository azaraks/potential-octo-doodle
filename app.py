import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from dotenv import load_dotenv


load_dotenv()
st.title("연습용 챗봇")

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
    st.session_state.chain = None

def start_chat():
    llm = ChatOpenAI(model_name='gpt-4o')
    st.session_state.chat_started = True
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("5천년 묵은 지혜를 가진 챗봇으로, 강건체, 문어체, 고어체로 답변합니다."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    st.session_state.chain = RunnableWithMessageHistory(prompt | llm,
                                                        lambda session_id: StreamlitChatMessageHistory(key=session_id),
                                                        input_messages_key="question",
                                                        history_messages_key="history",
                                                        )

if "messages" not in st.session_state or st.session_state.chain is None:
    start_chat()
else:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        with st.chat_message(role):
            st.markdown(message.content)

if "classifier" not in st.session_state:
    st.session_state.classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")

def get_classified(text):
    sentiment_result = st.session_state.classifier(text)[0]
    sentiment_pred, sentiment_score = sentiment_result["label"], sentiment_result["score"]
    sentiment_dict = {
        "LABEL_0": "긍정적",
        "LABEL_1": "부정적",
    }
    sentiment_pred = sentiment_dict[sentiment_pred]
    prompt_text = text + f"\n\n(감정분석 결과: {sentiment_pred}, {sentiment_score * 100:.2f}%)"
    return prompt_text

if prompt := st.chat_input():
    with st.chat_message("user"):
        st.markdown(get_classified(prompt))

    with st.chat_message("assistant"):
        config = {"configurable": {"session_id": "messages"}}
        result = st.session_state.chain.invoke({"question": prompt}, config=config)
        st.markdown(get_classified(result.content))
