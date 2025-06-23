#Imports
import boto3
from langchain_aws import ChatBedrock


#Create the bedrock client
boto3_client = boto3.client('bedrock-runtime')

#setting model inference parameters
inference_modifier = {
    "temperature": 0.5,
    "max_tokens": 1000,
    "top_p": 1,
}

#Create the llm
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=boto3_client,
    model_kwargs=inference_modifier
)

# Format the prompt as a message list
messages = [
    {
        "role": "user",
        "content": "Write an email from Mark, Hiring Manager, welcoming a new employee 'John Doe' to the company on his first day."
    }
]

# Generate the response
response = llm.invoke(messages)

# Display the result
print(response.content)

