import os
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_complete, gpt_4o_mini_complete, openai_embed
import gradio as gr
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

def get_file_list(dir_path):
    
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_type = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)
            file_list.append({
                "file_path": file_path,
                "file_type": file_type,
                "file_name": file_name
            })
    return file_list

def extract_text_from_file(file_path):
    print(f"Extracting text from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        # Try to decode with error handling
        text = content.decode('utf-8', errors='ignore')
        return text
    except Exception as e:
        print(f"Failed to read file {file_path} with any encoding: {e}")
        return f"Error reading file: {file_path}"
    

load_dotenv(override=True)
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")


base_dir = "/Users/harish/Harish_MAC/Learning/Projects/GenAI/unstructured_RAG"

working_dir =os.path.join(base_dir, "working_dir")

if not os.path.exists(working_dir):
  os.mkdir(working_dir)

lightrag = LightRAG(
    working_dir = working_dir,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
    chunk_token_size=300,
    chunk_overlap_token_size=50,
    log_level="DEBUG"
)
dir_path = os.path.join(base_dir, "knowledge-base")
print(dir_path)

file_list = get_file_list(dir_path)

for file in file_list:
  print(f"Inserting {file['file_path']} into the database...")
  with open(file["file_path"], "r", encoding="utf-8") as f:
    lightrag.insert(f.read())

query = "How many products does Insurellm have?"

result = lightrag.query(
               query,
               param=QueryParam(mode="naive")
               )
print(result)
