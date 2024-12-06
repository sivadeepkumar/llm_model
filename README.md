# Flask-based NLP Web Services

This repository provides Flask-based web services for various Natural Language Processing (NLP) tasks. Each service is built to leverage different AI/ML models and APIs to process text, provide intelligent responses, and perform document search and embedding operations.

## Features

- Integration with **LLama Index** for efficient document querying.
- Conversational AI using **OpenAI GPT-3.5**.
- Sentence embeddings via **Hugging Face Sentence Transformers**.
- Text processing and NLP services with **Amazon Comprehend**.
- Support for custom LLMs like **Mistral** and **Llama3**, powered by Amazon Bedrock.

## Installation and Setup

Follow these steps to set up and run the project:

### 1. Install Dependencies

Install the necessary Python libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Set up the required environment variables for model credentials and configurations:

export LANGCHAIN_API_KEY="your_langchain_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export MODEL_BIN="mistral-7b-instruct-v0.1.Q4_0"  # Example: "gpt4all-falcon-newbpe-q4_0"
export kb_id="your_kb_id"

3. Run the Application
Execute the main Flask application:

'''python app.py'''


4. Update Vector Database (Required for Document Search)
To initialize and update the vector database with your sample_data, make a GET request to the /update_vector_base endpoint:
'''
curl -X GET http://localhost:5000/update_vector_base
'''

Endpoints
1. Health Check
Endpoint: /health
Method: GET
Description: Check the health status of the application.
Response:{
  "status": "OK"
}


2. Mistral Model Response
Endpoint: /mistral_response
Method: POST
Description: Get responses from the Mistral model using user queries.
Request Body:{
  "query": "Your question here"
}
Response:
{
  "status": "Success",
  "Response": "Model response here"
}

3. Llama3 Model Response
Endpoint: /llama_response
Method: POST
Description: Get responses from the Llama3 model using user queries.
Request Body:

json
Copy code
{
  "query": "Your question here"
}
Response:

json
Copy code
{
  "status": "Success",
  "Response": "Model response here"
}
4. Bedrock Knowledge Base
Endpoint: /bedrock
Method: POST
Description: Query a specific knowledge base using Amazon Bedrock.
Request Body:

json
Copy code
{
  "query": "Your question here"
}
Response:

json
Copy code
{
  "status": "Success",
  "Response": "Knowledge base response here"
}
5. Custom Sources
Llama3 Model with Source
Endpoint: /llama/source
Method: POST
Description: Query Llama3 with a custom source.
Request Body:

json
Copy code
{
  "query": "Your question here",
  "source": "Custom text source here"
}
Mistral Model with Source
Endpoint: /mistral/source
Method: POST
Description: Query Mistral with a custom source.
Request Body:

json
Copy code
{
  "query": "Your question here",
  "source": "Custom text source here"
}
Project Structure
bash
Copy code
.
├── app.py                  # Main application file
├── requirements.txt        # Dependencies
├── sample_data/            # Directory containing sample PDFs and indexes
├── helper.py               # Utility functions for Bedrock agent
├── .env                    # Environment variables (optional)
└── README.md               # Documentation
Requirements
Python 3.8+
AWS credentials for accessing Amazon services
API keys for LangChain and OpenAI
Local or cloud GPU for large model inference
Notes
Always update the vector database with the /update_vector_base endpoint before querying.
Configure the .env file for storing sensitive API keys and model configurations securely.
Ensure AWS credentials are correctly set up for accessing Amazon Comprehend and Bedrock services.
