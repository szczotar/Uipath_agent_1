
from pydantic import BaseModel, Field
from typing import List, Optional, Sequence, Annotated
from uipath import UiPath
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode # Używamy ToolNode zgodnie z wymaganiem
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
# --- 1. Inicjalizacja i konfiguracja ---
sdk = UiPath()

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

tools = [download_file_from_uipath_bucket]

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

