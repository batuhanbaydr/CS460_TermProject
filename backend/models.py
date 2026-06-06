import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()


def get_bedrock_model(model_id: str, temperature: float = 0.2, max_tokens: int = 800):
    """
    Creates a Bedrock chat model using the given model ID.
    """
    return ChatBedrockConverse(
        model=model_id,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_test_models():
    """
    Returns the two models that we are testing the user's prompt on.
    These are intentionally lightweight models.
    """
    return {
        "amazon_nova_lite": get_bedrock_model(
            "amazon.nova-lite-v1:0",
            temperature=0.2,
            max_tokens=800,
        ),
        "meta_llama_3_8b_instruct": get_bedrock_model(
            "meta.llama3-8b-instruct-v1:0",
            temperature=0.2,
            max_tokens=800,
        ),
    }


def get_judge_model():
    """
    Returns the stronger model used for evaluation and prompt optimization.
    This model is not one of the tested models, which reduces bias.
    """
    return get_bedrock_model(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.0,
        max_tokens=1500,
    )


# Backward compatibility:
# Some older files may still call get_models(). For now, we make it return the test models
def get_models():
    return get_test_models()