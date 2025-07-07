import os
import shutil
import asyncio
import nest_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Apply nest_asyncio to handle async operations in Jupyter
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
        text = content.decode('utf-8', errors='ignore')
        return text
    except Exception as e:
        print(f"Failed to read file {file_path} with any encoding: {e}")
        return f"Error reading file: {file_path}"

async def main():
    # Clean up any existing working directory
    base_dir = "/Users/harish/Harish_MAC/Learning/Projects/GenAI/unstructured_RAG"
    working_dir = os.path.join(base_dir, "test_v135_working_dir")
    
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    
    print(f"Testing LightRAG version 1.3.5...")
    print(f"Working directory: {working_dir}")
    
    try:
        # Initialize LightRAG with version 1.3.5
        lightrag = LightRAG(
            working_dir=working_dir,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=openai_embed
        )
        
        print("✅ LightRAG 1.3.5 initialized successfully!")
        
        # Process a single test file
        dir_path = os.path.join(base_dir, "knowledge-base")
        file_list = get_file_list(dir_path)
        
        if file_list:
            test_file = file_list[0]  # Use first file
            print(f"Processing test file: {test_file['file_name']}")
            
            try:
                file_text = extract_text_from_file(test_file["file_path"])
                print(f"Text length: {len(file_text)} characters")
                
                # Try to insert the file
                print("Attempting to insert file...")
                lightrag.insert(file_text, file_paths=test_file["file_path"])
                print(f"✅ Successfully inserted {test_file['file_name']}")
                
                # Test query
                query = "What is Rellm?"
                print(f"\nTesting query: {query}")
                
                result = lightrag.query(query, param=QueryParam(mode="naive"))
                print("Query result:")
                print(result)
                
                print("\n🎉 SUCCESS! LightRAG 1.3.5 works without the storage lock issue!")
                
            except Exception as e:
                print(f"❌ Error during processing: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No files found in knowledge-base directory")
            
    except Exception as e:
        print(f"❌ Error initializing LightRAG 1.3.5: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 