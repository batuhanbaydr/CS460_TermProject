import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()


def get_bedrock_model(model_id: str):
    return ChatBedrockConverse(
        model=model_id,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        temperature=0.2,
        max_tokens=800,
    )


def get_models():
    return {
        "amazon_nova_lite": get_bedrock_model("amazon.nova-lite-v1:0"),
        "meta_llama_3_8b_instruct": get_bedrock_model("meta.llama3-8b-instruct-v1:0"),
    }