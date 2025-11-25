
from pydantic import BaseModel, Field
from typing import List, Optional, Sequence, Annotated
from uipath import UiPath
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode # Używamy ToolNode zgodnie z wymaganiem
from langchain_openai import ChatOpenAI
import os

import requests
import json
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
# --- 1. Inicjalizacja i konfiguracja ---
sdk = UiPath()

TRAFFIT_BASE_URL = "https://mindboxgroup.traffit.com"

def _get_traffit_token(base_url):
    """Helper function to get the authentication token."""
    url_endpoint = f"{base_url}/oauth2/token"
    scope = "employee, file, advert, advert_publish, client, crm_activity, crm_person, form, message, provision, recruitment, talent, user, webhook, workflow, source, dictionary"
    headers = {"Content-Type": "application/json"}
    body = {
        "client_id": "mindboxgroup_CVtagv2",
        "client_secret": os.getenv("Traffit_Client_Secret"),
        "grant_type": "client_credentials",
        "scope": scope.replace(",", "")
    }
    response = requests.post(url=url_endpoint, headers=headers, data=json.dumps(body))
    response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
    return response.json()['access_token']

def _get_candidate_data(base_url, candidate_id, access_token):
    """Helper function to get candidate data."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    url_endpoint = f"{base_url}/api/integration/v2/employees/{candidate_id}"
    response = requests.get(url=url_endpoint, headers=headers)
    response.raise_for_status()
    return response.json()
# --- 2. Definicja narzędzi (Tools) ---

@tool
async def download_file_from_uipath_bucket(file_path: str, bucket_name: str = "GenAI", orchestrator_folder: str = "Shared") -> str:
    """
    Downloads a file from a specified UiPath Storage Bucket and returns its content.
    Use this tool when the user asks to read, get, or download a file.
    
    Args:
        file_path (str): The name of the file to download (e.g., 'test.txt').
        bucket_name (str): The name of the storage bucket. Defaults to 'GenAI'.
        orchestrator_folder (str): The folder path in Orchestrator. Defaults to 'Shared'.
    """
    print(f"Narzędzie: Pobieranie pliku '{file_path}' z bucketa '{bucket_name}'...")
    local_temp_path = f"./{file_path}"
    try:
        await sdk.buckets.download_async(
            name=bucket_name,
            blob_file_path=file_path,
            destination_path=local_temp_path,
            folder_path=orchestrator_folder
        )
        print(f"Plik pomyślnie pobrany do '{local_temp_path}'.")
        with open(local_temp_path, "r", encoding='utf-8') as f:
            content = f.read()
        print("Zawartość pliku odczytana.")
        return f"Successfully read file '{file_path}'. Content: {content}"
    except Exception as e:
        error_message = f"Error downloading file: {e}"
        print(error_message)
        return error_message
    finally:
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)

@tool
def get_traffit_candidate_data(candidate_id: int) -> str:
    """
    Fetches data for a specific candidate from the Traffit system using their ID.
    Use this tool when the user asks to find, get, or search for a candidate in Traffit.
    
    Args:
        candidate_id (int): The numerical ID of the candidate to search for.
    """
    print(f"Narzędzie: Pobieranie danych kandydata o ID: {candidate_id} z Traffit...")
    try:
        token = _get_traffit_token(TRAFFIT_BASE_URL)
        candidate_data = _get_candidate_data(TRAFFIT_BASE_URL, candidate_id, token)
        # Zwracamy dane jako sformatowany string JSON, aby LLM mógł je łatwo przeczytać
        return json.dumps(candidate_data, indent=2, ensure_ascii=False)
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - Response: {http_err.response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    
tools = [download_file_from_uipath_bucket, get_traffit_candidate_data]

# --- 3. Definicja stanu (State) i wejścia/wyjścia ---

class Input(BaseModel):
    input_message: str

class State(BaseModel):
    input_message: Optional[str] = None
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Output(BaseModel):
    result: str

# --- 4. Konfiguracja LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY") , temperature=0).bind_tools(tools)

# --- 5. Definicja węzłów grafu (Nodes) ---

async def prepare_input_node(state: State) -> State:
    print("Węzeł: Przygotowywanie wejścia...")
    if state.input_message:
        state.messages.append(HumanMessage(content=state.input_message))
        state.input_message = None
    return state

async def call_model(state: State) -> State:
    print("Węzeł: Wywoływanie modelu LLM...")
    print(f'--{state}--')
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state.messages)
    print(f'--{response}--')
    state.messages.append(response)
    # print(response)
    print(f'--{state.messages}--')
    return state

# KLUCZOWA ZMIANA: Własny węzeł do wywoływania ToolNode
tool_node= ToolNode(tools)
# async def call_tools_and_preserve_history(state: State) -> State:
#     """
#     Wywołuje ToolNode, ale dba o zachowanie pełnej historii wiadomości.
#     """
#     print("Węzeł: Wykonywanie narzędzi z zachowaniem historii...")
#     # Wywołujemy ToolNode, ale tylko na ostatniej wiadomości (tej z tool_calls)
#     # To zwróci nam listę z samymi ToolMessage
#     tool_messages = await tool_node_executor.ainvoke(state.messages[-1])
    
#     # Dodajemy wyniki do PEŁNEJ historii wiadomości
#     state.messages.extend(tool_messages)
    
#     return state

def should_continue(state: State) -> str:
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

async def output_node(state: State) -> Output:
    final_message = state.messages[-1].content
    return Output(result=final_message)

# --- 6. Budowa i kompilacja grafu ---

builder = StateGraph(State, input=Input, output=Output)

# Dodajemy nasz nowy węzeł narzędziowy
builder.add_node("prepare_input", prepare_input_node)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node) # Używamy naszego wrappera
builder.add_node("output", output_node)

builder.add_edge(START, "prepare_input")
builder.add_edge("prepare_input", "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "output",
    },
)

builder.add_edge("tools", "agent")
builder.add_edge("output", END)

graph = builder.compile()


# from PIL import Image
# img = graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(img)
# img = Image.open("graph.png")
# img.show()

