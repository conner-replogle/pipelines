"""
title: Google GenAI Manifold Pipeline
author: Marc Lopez (refactor by justinh-rahb)
date: 2024-06-06
version: 1.3
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: google-genai
environment_variables: GOOGLE_API_KEY
"""

from typing import List, Union, Iterator
import os

from pydantic import BaseModel, Field

from google import genai
from google.genai import types
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
        if (self.valves.GOOGLE_API_KEY):
            self.client = genai.Client(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        self.client = genai.Client(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""

        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""

        print(f"on_valves_updated:{__name__}")
        self.client = genai.Client(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    def update_pipelines(self) -> None:
        """Update the available models from Google GenAI"""

        if self.valves.GOOGLE_API_KEY:
            try:
                models = self.client.models.list()
        
                self.pipelines = [
                    {
                        "id": model.name[7:],  # the "models/" part messeses up the URL
                        "name": model.display_name,
                    }
                   
                    for model in models
                    if "generateContent" in model.supported_actions
                    if model.name[:7] == "models/"
                ]
            except Exception:
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
            self.client = genai.Client(api_key=self.valves.GOOGLE_API_KEY)
            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")

            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            
            contents = []
            for message in messages:
                if message["role"] != "system":
                    if isinstance(message.get("content"), list):
                        parts = []
                        for content in message["content"]:
                            if content["type"] == "text":
                                parts.append(types.Part.from_text(content["text"]))
                            elif content["type"] == "image_url":
                                image_url = content["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    image_data = image_url.split(",")[1]
                                    parts.append(types.Part.from_uri(image_data, mime_type="image/jpeg"))
                                else:
                                    parts.append(types.Part.from_uri(image_url))
                        contents.append(types.Content(parts,role="user" if message["role"] == "user" else "model"))
                    else:
                        contents.append(types.Content([types.Part.from_text(message["content"])],role="user" if message["role"] == "user" else "model"))

        

            generation_config = types.GenerateContentConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
                system_instruction=system_message,
                tools=[types.Tool(
                    google_search=types.GoogleSearchRetrieval
                )]
            )

            if body.get("stream", False):

                response = self.client.models.generate_content_stream(
                    model=model_id,
                    contents=contents,
                    config=generation_config,
                )
                return self.stream_response(response)
            else:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=generation_config,
                )
                return response.text


    

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            print(chunk)
            if chunk.text:
                sources = ""
                if chunk.candidates[0].grounding_metadata is not None and  chunk.candidates[0].grounding_metadata.grounding_chunks is not None:
                    for grounding_chunk in chunk.candidates[0].grounding_metadata.grounding_chunks:
                        sources += f"\nSource: [{grounding_chunk.web.title}]({grounding_chunk.web.uri})"
                    
                yield chunk.text + sources
