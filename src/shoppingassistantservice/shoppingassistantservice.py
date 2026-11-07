#!/usr/bin/python
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://apache.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import MagicMock
from urllib.parse import unquote
from langchain_core.messages import HumanMessage
from flask import Flask, request

# =====================================================================
# OFFLINE LOCAL MOCK CLASSES
# =====================================================================
class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
        
    def to_json(self):
        return {"page_content": self.page_content, "metadata": self.metadata}

class FakeVectorStore:
    def similarity_search(self, query, k=3):
        return [
            FakeDocument("A beautiful minimalist oak coffee table.", {"id": "prod-101", "name": "Oak Table", "categories": "furniture"}),
            FakeDocument("An elegant modern ceramic desk lamp.", {"id": "prod-102", "name": "Ceramic Lamp", "categories": "lighting"}),
            FakeDocument("A soft bohemian style tufted area rug.", {"id": "prod-103", "name": "Boho Rug", "categories": "decor"})
        ]

class FakeLLMResponse:
    def __init__(self, content):
        self.content = content

class FakeChatModel:
    def __init__(self, model):
        self.model = model
        
    def invoke(self, messages_or_prompt):
        # If it looks like the final design prompt, return the final recommendation string
        if isinstance(messages_or_prompt, str) and "You are an interior designer" in messages_or_prompt:
            return FakeLLMResponse("Based on the room layout, I recommend incorporating clean wooden textures and cozy floor accents. Here are the top items from the catalog: [prod-101], [prod-102], [prod-103]")
        # Otherwise, assume it is Step 1 (the vision model room description)
        return FakeLLMResponse("A modern minimalist room featuring clean white walls, warm natural wooden accents, low-profile contemporary furniture, and abundant ambient natural lighting.")

# =====================================================================
# ENVIRONMENT CONFIGURATION WITH SAFE FALLBACKS
# =====================================================================
PROJECT_ID = os.environ.get("PROJECT_ID", "local-testing")
REGION = os.environ.get("REGION", "us-central1")
ALLOYDB_DATABASE_NAME = os.environ.get("ALLOYDB_DATABASE_NAME", "local-db")
ALLOYDB_TABLE_NAME = os.environ.get("ALLOYDB_TABLE_NAME", "local-table")
ALLOYDB_CLUSTER_NAME = os.environ.get("ALLOYDB_CLUSTER_NAME", "local-cluster")
ALLOYDB_INSTANCE_NAME = os.environ.get("ALLOYDB_INSTANCE_NAME", "local-instance")
ALLOYDB_SECRET_NAME = os.environ.get("ALLOYDB_SECRET_NAME", "local-secret")

# The crucial switch flag. Set to "true" inside Docker to run purely offline.
LOCAL_DEMO = os.environ.get("LOCAL_DEMO", "true").lower() == "true"

# =====================================================================
# CONDITIONAL INITIALIZATION (REAL CLOUD VS LOCAL MOCK)
# =====================================================================
if not LOCAL_DEMO:
    # This block will ONLY run if you explicitly set LOCAL_DEMO=false
    from google.cloud import secretmanager_v1
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

    secret_manager_client = secretmanager_v1.SecretManagerServiceClient()
    secret_name = secret_manager_client.secret_version_path(project=PROJECT_ID, secret=ALLOYDB_SECRET_NAME, secret_version="latest")
    secret_request = secretmanager_v1.AccessSecretVersionRequest(name=secret_name)
    secret_response = secret_manager_client.access_secret_version(request=secret_request)
    PGPASSWORD = secret_response.payload.data.decode("UTF-8").strip()

    engine = AlloyDBEngine.from_instance(
        project_id=PROJECT_ID,
        region=REGION,
        cluster=ALLOYDB_CLUSTER_NAME,
        instance=ALLOYDB_INSTANCE_NAME,
        database=ALLOYDB_DATABASE_NAME,
        user="postgres",
        password=PGPASSWORD
    )

    vectorstore = AlloyDBVectorStore.create_sync(
        engine=engine,
        table_name=ALLOYDB_TABLE_NAME,
        embedding_service=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        id_column="id",
        content_column="description",
        embedding_column="product_embedding",
        metadata_columns=["id", "name", "categories"]
    )
else:
    # Bypasses all Google APIs completely
    print("--- RUNNING IN OFFLINE LOCAL MOCK MODE ---")
    PGPASSWORD = "local-password-mock"
    engine = MagicMock()
    vectorstore = FakeVectorStore()
    ChatGoogleGenerativeAI = FakeChatModel
def create_app():
    app = Flask(__name__)

    @app.route("/", methods=['POST'])
    def talkToGemini():
        print("Beginning RAG call")
        prompt = unquote(request.json['message'])

        # Step 1 – Get a room description from Gemini model proxy
        llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are a professional interior designer, give me a detailed description of the style of the room in this image",
                },
                {"type": "image_url", "image_url": request.json['image']},
            ]
        )
        response = llm_vision.invoke([message])
        description_response = response.content

        # Step 2 – Similarity search with the description & user prompt
        vector_search_prompt = f"This is the user's request: {prompt} Find the most relevant items for that prompt, while matching style of the room described here: {description_response}"
        docs = vectorstore.similarity_search(vector_search_prompt)
        
        relevant_docs = ""
        for doc in docs:
            doc_details = doc.to_json() if hasattr(doc, 'to_json') else str(doc)
            relevant_docs += str(doc_details) + ", "

        # Step 3 – Tie it all together by augmenting our call to Gemini model proxy
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        design_prompt = (
            f"You are an interior designer that works for Online Boutique... "
            f"This is the description of the room: \n{description_response} "
            f"Here are a list of products that are relevant to it: {relevant_docs} "
            f"Specifically, this is what the customer has asked for: {prompt}"
        )
        design_response = llm.invoke(design_prompt)
        return {'content': design_response.content}

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=8080)
