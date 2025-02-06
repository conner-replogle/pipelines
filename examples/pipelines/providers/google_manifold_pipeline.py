"""
title: Google GenAI Manifold Pipeline
author: Marc Lopez (refactor by justinh-rahb)
date: 2024-06-06
version: 1.3
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: python-genai
environment_variables: GOOGLE_API_KEY
"""
from typing import List, Union, Iterator
import os
from pydantic import BaseModel, Field
from genai import Client, Model, GenerateParams
from genai.exceptions import GenAiException
from genai.schemas import ModelType

class Pipeline:
    """Google GenAI pipeline"""
    class Valves(BaseModel):
        """Options to change from the WebUI"""
        GOOGLE_API_KEY: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.id = "google_genai"
        self.name = "Google: "
        self.valves = self.Valves(**{
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "USE_PERMISSIVE_SAFETY": False
        })
        self.pipelines = []
        self.client = Client(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_startup(self) -> None:
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        self.client = Client(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        self.client = Client(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    def update_pipelines(self) -> None:
        """Update the available models from Google GenAI"""
        if self.valves.GOOGLE_API_KEY:
            try:
                models = self.client.list_models()
                self.pipelines = [
                    {
                        "id": model.name,
                        "name": model.display_name,
                    }
                    for model in models
                    if model.supports_generation
                ]
            except GenAiException:
                self.pipelines = [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Google, please update the API Key in the valves.",
                    }
                ]
        else:
            self.pipelines = []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"
        
        try:
            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")
            
            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")

            model = Model(model_id, client=self.client)
            
            # Process messages
            conversation = []
            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            
            if system_message:
                conversation.append({"role": "system", "content": system_message})
            
            for message in messages:
                if message["role"] != "system":
                    if isinstance(message.get("content"), list):
                        # Handle multimodal content
                        content = []
                        for part in message["content"]:
                            if part["type"] == "text":
                                content.append({"type": "text", "text": part["text"]})
                            elif part["type"] == "image_url":
                                image_url = part["image_url"]["url"]
                                content.append({"type": "image", "source": image_url})
                        conversation.append({"role": message["role"], "content": content})
                    else:
                        conversation.append({
                            "role": message["role"],
                            "content": message["content"]
                        })

            # Configure generation parameters
            params = GenerateParams(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
                stream=body.get("stream", False)
            )

            if self.valves.USE_PERMISSIVE_SAFETY:
                params.safety_settings = {
                    "harassment": "none",
                    "hate_speech": "none",
                    "sexually_explicit": "none",
                    "dangerous_content": "none"
                }

            response = model.generate(
                messages=conversation,
                params=params
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except GenAiException as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                yield chunk.text