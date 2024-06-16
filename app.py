from langchain.memory import ConversationBufferMemory 
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
import streamlit as st
from langchain.memory import ChatMessageHistory
from custom_tools import search_bible

# page config
st.set_page_config(page_title="BibleBot",
                   page_icon=':sparkles:',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.title(':latin_cross: BibleBot')

# Set up open ai key
openai_key = st.secrets["OPEN_AI_KEY"]

# Initialise chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def initalise_agent():
# Add memory to the chat for conversation history
formatted_chat_history = ChatMessageHistory()
for message in st.session_state.chat_history:
    if message['role']=='user':
        formatted_chat_history.add_user_message(message['content'])
    elif message['role']=='assistant':
        formatted_chat_history.add_ai_message(message['content'])
memory = ConversationBufferMemory(chat_memory=formatted_chat_history,
                                    return_messages=True,
                                    memory_key="chat_history",
                                    output_key="output")


# Create and format tools to be able to be used by open ai
tools = [search_bible]
functions = [format_tool_to_openai_function(f) for f in tools]

# load Agent model from open AI
llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0125",
                    openai_api_key = openai_key,
                    temperature = 0.2, 
                    streaming=True)

# Bind the tools to the model
model = llm.bind(functions=functions)

# Create prompt using the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly chatbot who has access to a tool that can search the bible and return the top 3 relevant texts. 
    Use the tool search_bible by giving it a query in the form of a string in order to answer the user query.
    Answer the question based on the texts that you get by using the tool. 
    If you are going to answer the question, mention the bible verse that you are talking about.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# create chain
chain = RunnablePassthrough.assign(
        agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()

# create executor
executor = AgentExecutor(agent=chain, 
                            tools=tools, 
                            verbose=False, 
                            return_intermediate_steps=True,
                            handle_parsing_errors=True,
                            memory=memory)

# ----- Retrieval and Generation Process -----



# this is an example of a user question (query)
query = 'What is that verse that talks about the heel crushing the head of a snake?'

# retrieve context - top 5 most relevant (closests) chunks to the query vector 
docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

# generate an answer based on given user query and retrieved context information
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

# you can use a prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don't give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

# load retrieved context and user query in the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# call LLM model to generate the answer based on the given context and query
model = ChatOpenAI(model_name = "gpt-3.5-turbo-0125",
                    openai_api_key = st.secrets['openai_api_key'],
                    temperature = 0.2, 
                    streaming=True)
response_text = model.predict(prompt)


# function to display chat
def display_chat():
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

# function to get model answer
def create_answer(question):
    # create answer
    result = executor.invoke({'input':question})
    # add that message too the chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["output"],
    })

# If we get a question, start doing stuff!
if question := st.chat_input(placeholder="Let's chat"):
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })
    create_answer(question)
    display_chat()