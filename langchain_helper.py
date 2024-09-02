"""
# LangChain Helper

LangChain code for using HuggingFace text-generation model as in LLM chain!!

This module provides a function `generate_pet_name` that uses a HuggingFace text-generation model to generate pet names based on animal type and color.
"""

from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
import os

load_dotenv()


# Prompt for LLM
CONSTANT_PROMPT = """
I have a {animal_type} pet and i want 5 cool and matching names for it, it's {pet_color} in color.
Suggest me 5 cool names for my pet, it's description : {animal_description}.
Just provide 5 ordered points formatted on separate lines, no description or extra text!
"""


def generate_pet_name(
    animal_type: str, animal_color: str, animal_description: str
) -> str:
    # HF Inference API as LLM
    llm = HuggingFaceEndpoint(
        repo_id=os.environ["HUGGINGFACE_MODEL_ID"],
        task="text-generation",
        max_new_tokens=50,
        repetition_penalty=1.03,
        temperature=0.8,
    )  # type: ignore

    # Prompt template
    prompt = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template=CONSTANT_PROMPT,
    )

    chain = prompt | llm  # chaining them!

    response = chain.invoke({
        "animal_type": animal_type,
        "pet_color": animal_color,
        "animal_description": animal_description,
    })

    return response


if __name__ == "__main__":
    response = generate_pet_name("ball python", "Green", "loves palm trees")
    print(response)


