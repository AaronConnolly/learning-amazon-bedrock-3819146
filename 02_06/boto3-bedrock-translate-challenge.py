#Imports
import boto3
import json

#Create the client
client = boto3.client(service_name='bedrock-runtime')

#Construct the body
prompt = "Learning about Generative AI is fun and exciting using Amazon Bedrock"
Task = "Translate the following text to French: "

#specify your prompt
body = json.dumps({
    "prompt": Task + prompt, 
    "temperature": 0.5,
})

#Specify model id and content types
modelId = 'mistral.mistral-large-2402-v1:0'
accept = 'application/json'
contentType = 'application/json'

#Invoke the model
response = client.invoke_model(
    body=body, 
    modelId=modelId, 
    accept=accept, 
    contentType=contentType
)

#Extract the response
response_body = json.loads(response.get('body').read())

#Display the output
print(response_body['outputs'][0]['text'])