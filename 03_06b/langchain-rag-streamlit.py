#Import libraries
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

#Configure streamlit app
st.set_page_config(page_title="Social Media Training Bot", page_icon="📖")
st.title("📖 Social Media Training Bot")

#Define convenience functions
@st.cache_resource
def config_llm():
    client = boto3.client('bedrock-runtime')

    model_kwargs = { 
        "max_tokens": 512,
        "temperature":0.1,  
        "top_p":1
    }  

    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    llm = ChatBedrock(
        model_id=model_id, 
        client=client,
        model_kwargs=model_kwargs
    )
    return llm

@st.cache_resource
def config_vector_db(filename):
    client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(
        client=client,
        model_id="amazon.titan-embed-text-v2:0"
    )
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

#Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db('03_04b/social-media-training.pdf')

#Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello can I help you?")
    

#Creating the template   
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from an employee. 
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Do not say things like "according to the training or handbook or based on or according to the information provided...".

<Information>
{info}
</Information>

{input}

Assistant:
"""

#Configure prompt template
prompt_template = PromptTemplate(
    input_variables=["input", "info"],
    template=my_template
)

#Create llm chain
question_chain = prompt_template | llm

#Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

#If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # Retrieve the most relevant documents from the vector store using similarity search
    docs = vectorstore_faiss.similarity_search_with_score(prompt)
    info = ""
    for doc in docs:
        info += doc[0].page_content + '\n'

    # Invoke the question chain with the prompt and the retrieved information
    output = question_chain.invoke({
        "input": prompt,
        "info": info
    })

    # Add the AI response to the chat message history
    msgs.add_user_message(prompt)
    msgs.add_ai_message(output.content)

    # Display the AI response in the chat
    st.chat_message("ai").write(output.content)