from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.youtube import YoutubeLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api.formatters import TextFormatter
from langchain.agents import initialize_agent, AgentType
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.callbacks import get_openai_callback
from langchain.prompts import MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import utils
import json
import os

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=3950, chunk_overlap=100)

def create_stuff_summary(docs):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='stuff',
        verbose=True,
    )
    return summary_chain.run(input_documents=docs)

def create_map_summary(docs):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    map_prompt_template = PromptTemplate(
        template="""
                Write a concise summary of the following part of an article or youtube video transcript. 
                If it contains a tutorial or guide, please include and summarize each step as well:
                "{text}"
                SUMMARY:
                """,
        input_variables=["text"]
    )

    combine_prompt_template = PromptTemplate(
        template="""
                Write a summary of the following text, which consists of summaries of parts from a whole article or youtube video transcript.
                Make it coherent and include step by step explanations, here comes the text:
                "{text}"
                SUMMARY:
                """,
        input_variables=["text"]
    )
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        verbose=True,
        map_prompt = map_prompt_template,
        combine_prompt = combine_prompt_template
    )

    return summary_chain.run(input_documents=docs)

def create_summary(docs):
    if len(docs) > 1:
        output = create_map_summary(docs)
    else:
        output = create_stuff_summary(docs)

    return output

@tool
def summarize_youtube_video(youtube_url: str) -> str:
    """
    Useful to get the summary of a youtube video. Applies if the user sends a youtube link.
    """
    id_input = youtube_url.split("=")[1]
    splitter = text_splitter
    loader = YoutubeLoader(id_input)
    docs = loader.load_and_split(splitter)
    return create_summary(docs)

def create_docs(text):
    docs = text_splitter.create_documents([text])
    return docs

@tool
def summarize_article(article_url: str) -> str:
    """
    Useful to summarize an online Article. Applies if the user sends an url which is not 
    a youtube link.
    """
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data = {"url": article_url}

    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={os.getenv('BROWSERLESS_TOKEN')}"
    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        docs = create_docs(text)
        return create_summary(docs)
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return "Sorry, I could not summarize this article, tell the user that there was an error"
    

class summary_agent():

    def __init__(self) -> None:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        memory = ConversationSummaryBufferMemory(
            memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

        system_message = SystemMessage(
        content="""You are a world class summarizer, who produces concise but full summaries of any topic based on youtube links or web articles links.
                The user will give you either a youtube video link or a web article link or questions, or a link with an additional question. 
                If it is a link, you decide which tool to use to produce a summary and then you return the summary, and
                also provide helpful and concise answers to questions the user gave you.
                """
        )

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_message,
        }

        tools = [summarize_youtube_video, summarize_article]

        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )

    def summarize(self, query):
        with get_openai_callback() as cb:
            output = self.agent.run(input=query)
            print(output)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")    
            utils.log_openai_usage(cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)
            return output
        
if __name__ == "__main__":
    print("input something...")
    agent = summary_agent()
    query = input()
    print(agent.summarize(query))
