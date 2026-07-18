import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

# Load Environment Variables
load_dotenv(override=True)

# Load Model
model = ChatMistralAI(model="mistral-small-2506")

# Page Title
st.set_page_config(page_title="Movie Information Extractor")
st.title("🎬 Movie Information Extractor")

st.write("Enter a movie paragraph and get structured information.")

# Pydantic Schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str

# Output Parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract movie information from the paragraph.

{format_instructions}
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])

# Text Input
paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=200,
    placeholder="Example: Inception is a 2010 science fiction film directed by Christopher Nolan..."
)

# Button
if st.button("Extract Movie Information"):

    if paragraph.strip() == "":
        st.warning("Please enter a movie paragraph.")
    else:

        final_prompt = prompt.invoke({
            "paragraph": paragraph,
            "format_instructions": parser.get_format_instructions()
        })

        with st.spinner("Extracting information..."):

            response = model.invoke(final_prompt)

        st.subheader("Raw Model Output")
        st.code(response.content)

        movie_data = parser.parse(response.content)

        st.subheader("Structured Output")
        st.json(movie_data.model_dump())