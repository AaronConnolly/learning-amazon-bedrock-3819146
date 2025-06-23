#Import libraries
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

#Define vectorstore
global vectorstore_faiss

#Define convenience functions
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

def vector_search (query):
    docs = vectorstore_faiss.similarity_search_with_score(query)
    info = ""
    for doc in docs:
        info+= doc[0].page_content+'\n'
    return info    


#Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db('03_04b/social-media-training.pdf')

#Creating the template   
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from an employee. 
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Only answer the question. Do not say things like "according to the training or handbook or according to the information provided...".
    
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

#Get question, peform similarity search, invoke model and return result
while True:
    question = input("\nAsk a question about the social media training manual:\n ")

    # Perform similarity search
    info = vector_search(question)

    # invoke model, providing additinal context
    output = question_chain.invoke({
        "input": question,
        "info": info
    })

    # display the result
    print(output.content if hasattr(output, "content") else output)
