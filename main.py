from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, Form
import os
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
# from langgraph.graph import Graph
import json
import requests
from langchain_core.messages import HumanMessage, convert_to_messages
import asyncio
import re
from langgraph_sdk import get_client
from form_config import form_config
from formatConfigForCalculator import format_config_for_calculator
from file_processor import process_file
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open('iscc.json', 'r') as f:
    input_data = json.load(f)

app = FastAPI(title="LangGraph Model Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for LangGraph models
# langgraph_models: Dict[str, Graph] = {}

class ModelConfig(BaseModel):
    name: str
    config: Dict[str, Any]

class UserInput(BaseModel):
    message: str
    thread_id: str = None

@app.get("/")
async def root():
    return {"message": "LangGraph Model Server is running"}


@app.post("/create_thread_guide")
async def create_thread_guide():
    """Create a new thread for the LangGraph model"""
    # url = 'http://localhost:8123'
    # url = 'http://iscc-eu-cal-guide-inline-context.railway.internal:8080'
    
    
    # url ='https://iscc-eu-cal-guide-inline-context-production.up.railway.app/'
    
    # url ='https://iscc-eu-cal-guide-inline-context-without-rag-production.up.railway.app/'
    # url='http://iscc-eu-cal-guide-inline-context-without-rag.railway.internal:8080'
    # Use public URL for Render deployment (can't access Railway internal network)
    url='https://iscc-eu-cal-guide-inline-context-without-rag-production.up.railway.app'
    client = get_client(url=url)
    thread = await client.threads.create()
    return {"thread_id": thread["thread_id"]}


@app.websocket("/chat_guide")
async def websocket_endpoint_guide(websocket: WebSocket):
    """WebSocket endpoint for interactive chat with the LangGraph model"""
    await websocket.accept()
    
    # url = 'http://localhost:8123'
    # url = 'http://iscc-eu-cal-guide-inline-context.railway.internal:8080'


    # url ='https://iscc-eu-cal-guide-inline-context-production.up.railway.app'
    # url ='https://iscc-eu-cal-guide-inline-context-without-rag-production.up.railway.app/'
    # url='http://iscc-eu-cal-guide-inline-context-without-rag.railway.internal:8080'
    # Use public URL for Render deployment (can't access Railway internal network)
    url='https://iscc-eu-cal-guide-inline-context-without-rag-production.up.railway.app'

    client = get_client(url=url)
    

    try : 
        # Loop to handle multiple messages in the same connection
        while True:
            raw_data = await websocket.receive_text()
            # print(f"Received data: {raw_data}")
            logger.info(f"Received data: {raw_data}")
            
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
                continue  # Don't close, wait for next message
            
            thread_id = data.get("thread_id")
            message = data.get("message")
            
            if not thread_id:
                await websocket.send_json({"error": "Thread ID is required"})
                continue  # Don't close, wait for next message

            if not message:
                await websocket.send_json({"error": "Message is required"})
                continue  # Don't close, wait for next message
            
            logger.info("HI")

            config = {'configurable': {
                                        "user_id": "123",
                                        # "response_model": "google_genai/gemini-2.0-flash",
                                        # "response_model": "openai/gpt-4.1-nano",
                                        "response_model": "openai/gpt-4.1-mini",
                                        # "response_model_advance": "openai/gpt-4.1-mini",
                                        "response_model_advance": "openai/gpt-4o",
                                        }
                      }
            
            graph_name = "Regulation_graph_without_rag"
            result = ""

            async for chunk in client.runs.stream(thread_id, 
                                            graph_name, 
                                            input={
                                                    "messages": [
                                                        {
                                                            "role": "user",
                                                            "content": message
                                                        }
                                                    ]
                                                },
                                            config=config,
                                            stream_mode="messages-tuple"):
                
                logger.info("==========Inside for loop==========")
                logger.info(f"chunk : {chunk}")
                
                if chunk.event == "messages":
                    result += "".join(data_item['content'] for data_item in chunk.data if 'content' in data_item)
            
            print("****"*25)
            logger.info(f"raw result : {result}")
            
            # Clean using the proven working logic: grab everything AFTER the last '}'
            cleaned_result = result.rpartition('}')[-1].strip()
            
            logger.info("****"*25)
            logger.info(f"cleaned result : {cleaned_result}")
            
            # FIX: If cleaned_result is empty (cold-start issue), retry once after waiting
            if not cleaned_result:
                logger.info("Empty response detected, retrying after 1 second...")
                await asyncio.sleep(1)  # Wait for LangGraph to warm up
                
                # Retry the stream
                result = ""
                async for chunk in client.runs.stream(thread_id, 
                                                graph_name, 
                                                input={
                                                        "messages": [
                                                            {
                                                                "role": "user",
                                                                "content": message
                                                            }
                                                        ]
                                                    },
                                                config=config,
                                                stream_mode="messages-tuple"):
                    if chunk.event == "messages":
                        result += "".join(data_item['content'] for data_item in chunk.data if 'content' in data_item)
                
                # Clean again
                cleaned_result = result.rpartition('}')[-1].strip()
                logger.info(f"retry cleaned result : {cleaned_result}")

            # Safety check: if still empty, send a helpful message
            if not cleaned_result:
                cleaned_result = "I'm warming up. Please try your question again in a moment."
            
            # Send the cleaned result once at the end
            await websocket.send_text(cleaned_result)
            
            # Signal end of this response
            await websocket.send_text("[DONE]")
                        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass  # Client already disconnected
    finally:
        try:
            await websocket.close()
        except:
            pass  # Already closed
        print("Task finished")






@app.post("/create_thread_information_extractor")
async def create_thread_information_extractor():
    """Create a new thread for the LangGraph model"""
    url = 'http://iscc-eu-information-extractor.railway.internal:8080'
    client = get_client(url=url)
    thread = await client.threads.create()
    return {"thread_id": thread["thread_id"]}



def process_name(name : str) -> str:
    name = name.replace(".", "_")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")

    if len(name) > 38:
        name = name[:30]
    
    return name


@app.websocket("/chat_information_extractor")
async def websocket_endpoint_information_extractor(websocket: WebSocket):
    """WebSocket endpoint for interactive chat with the LangGraph model"""
    await websocket.accept()
    
    url = 'http://iscc-eu-information-extractor.railway.internal:8080'
    client = get_client(url=url)
    

    try : 
        data = await websocket.receive_text()
        data = json.loads(data)
        
        thread_id = data.get("thread_id")
        message = data.get("message")
        pdf_name = data.get("pdf_name")
        pdf_name = process_name(pdf_name)
        
        if not thread_id:
            await websocket.send_json({"error": "Thread ID is required"})
            

        config = {'configurable': {
            "user_id": "123",
            "pdf_name": pdf_name,
            "response_model": "openai/gpt-4o",
            # "query_model": "openai/gpt-4o",
            "query_model": "openai/gpt-4.1-mini",
            # "query_model": "google_genai/gemini-2.0-flash",
            }
            }
        graph_name = "retrieval_graph"

        result = ""
        async for chunk in client.runs.stream(thread_id, 
                                        graph_name, 
                                        input={"messages": [HumanMessage(content=message)]},
                                        config=config,
                                        # interrupt_before=["More_Info"],
                                        stream_mode="messages-tuple"):
            if chunk.event == "messages":
                result += "".join(data_item['content'] for data_item in chunk.data if 'content' in data_item)
        
            
        cleaned_message = re.sub(r'^\s*\{"logic":.*?\}\s*', '', result, flags=re.DOTALL)
        cleaned_message = re.sub(r'^\s*\{"steps":.*?\}\s*', '', cleaned_message, flags=re.DOTALL)
        cleaned_message = re.sub(r'^\s*\{"queries":.*?\}\s*', '', cleaned_message, flags=re.DOTALL)
        cleaned_message = re.sub(r'^\s*\{"queries":.*?\}\s*', '', cleaned_message, flags=re.DOTALL)
        cleaned_message = re.sub(r'^\s*\{"queries":.*?\}\s*', '', cleaned_message, flags=re.DOTALL)

        for i in range(0, len(cleaned_message), 3):
            chunk = cleaned_message[i:i+3]
            await websocket.send_text(chunk)
            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()



    



# Report Agent
def extract_formatted_document(text):
    # Look for the pattern that starts with "formatted_document': " followed by content
    pattern = r"formatted_document':\s*\"(.*?)(?:\"\s*,|\"\s*$|\]$)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Extract the content and handle escape characters
        content = match.group(1)
        # Replace escaped newlines with actual newlines if needed
        content = content.replace("\\n", "\n")
        return content
    else:
        # Try an alternative pattern if the first one fails
        alt_pattern = r"formatted_document':\s*(.*?)(?:,\s*\w+:|$|\])"
        alt_match = re.search(alt_pattern, text, re.DOTALL)
        if alt_match:
            content = alt_match.group(1).strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            content = content.replace("\\n", "\n")
            return content
        
        # If all else fails, try to find the section by looking for the markdown header
        header_pattern = r"# Greenhouse Gas Intensity of Corn Production(.*?)(?:\]$|\Z)"
        header_match = re.search(header_pattern, text, re.DOTALL)
        if header_match:
            return "# Greenhouse Gas Intensity of Corn Production" + header_match.group(1)
            
        return "Formatted document section not found"



def extract_edited_section(text):
    # Look for the pattern that starts with "edited': " or "edited: " followed by content
    pattern = r"edited'?:\s*\"?(.*?)(?:\"?\s*,|\"\s*$|\]$|$)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Extract the content and handle escape characters
        content = match.group(1).strip()
        # Remove surrounding quotes if present
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        # Replace escaped newlines with actual newlines if needed
        content = content.replace("\\n", "\n")
        return content
    else:
        # Try an alternative pattern if the first one fails
        alt_pattern = r"edited'?:\s*(.*?)(?:,\s*\w+:|$|\])"
        alt_match = re.search(alt_pattern, text, re.DOTALL)
        if alt_match:
            content = alt_match.group(1).strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            content = content.replace("\\n", "\n")
            return content
        
        # If all else fails, try to find the section by looking for the field directly
        field_pattern = r"edited'?:\s*(.*?)(?:human_messages|\Z)"
        field_match = re.search(field_pattern, text, re.DOTALL)
        if field_match:
            content = field_match.group(1).strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content
            
        return "Edited section not found"

@app.post("/create_thread_training")
async def create_thread_training(input_json:Optional[Dict[str,Any]]=None):
    url_for_train_deployment = "training_agent.railway.internal"
    client = get_client(url = url_for_train_deployment)
    thread = await client.threads.create()
    config ={"configurable":{"user_id":"Test"}}
    graph_name = "training_agent"
    result=""
    json_op={}
    async for chunk in client.runs.stream(
        thread["thread_id"],
        graph_name,
        input={"messages": [HumanMessage(content="Extract all the information.")] },
        config=config,
        stream_mode="values"
    ):
        print(chunk)
        if chunk.event=="values":
            json_op = chunk.data
    return json_op



@app.post("/create_thread_report")
async def create_thread_report(input_json:Optional[Dict[str,Any]]=None):
    url_for_cli_deployment = "http://iscc-eu-report.railway.internal:8080"
    client = get_client(url=url_for_cli_deployment)
    thread = await client.threads.create()
    if input_json:
        thread_data[thread["thread_id"]]=input_json
    return {"thread_id": thread["thread_id"]}                   
thread_data={}
async def handle_initial_message_report(websocket: WebSocket, thread_id: str,input_json: Dict[str, Any]):
    url_for_cli_deployment = "http://iscc-eu-report.railway.internal:8080"
    client = get_client(url=url_for_cli_deployment)
    config = {"configurable": {"user_id": "Test"}}
    graph_name = "retrieval_graph"
    
    result = ""
    async for chunk in client.runs.stream(
        thread_id,
        graph_name,
        input={"input_data":input_json },#change the data here to input_json
        config=config,
        interrupt_before=["human_edit"],
        stream_mode="messages-tuple"
    ):
        
        logger.info(f"=========inside handel initial message report=========")
        logger.info(f"chunk : {chunk}")
        if chunk.event == "messages":
            result += "".join(data_item['content'] for data_item in chunk.data if 'content' in data_item)

    formatted_document = extract_formatted_document(result)
    print(formatted_document)
    for i in range(0, len(formatted_document), 3):
        chunk = formatted_document[i:i+3]
        await websocket.send_text(chunk)
        await asyncio.sleep(0.008)

@app.websocket("/route_for_chat_report")
async def websocket_endpoint_report(websocket: WebSocket):
    print("WebSocket connection attempt...")
    await websocket.accept()
    print("WebSocket connection accepted")
    initial_message_handled = False
    url_for_cli_deployment = "http://iscc-eu-report.railway.internal:8080"
    client = get_client(url=url_for_cli_deployment)
    print(f"LangGraph client initialized for URL: {url_for_cli_deployment}")
    session_input_json = None
    try:
        while True:
            print("Waiting for client message...")
            # Receive message from client
            try:
                data_raw = await websocket.receive_text()
                print(f"Received raw data: {data_raw}")
                data = json.loads(data_raw)
                print(f"Parsed JSON data: {data}")
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error: {str(e)}"
                print(error_msg)
                await websocket.send_text(error_msg)
                continue
            except Exception as e:
                error_msg = f"Error receiving message: {str(e)}"
                print(error_msg)
                await websocket.send_text(error_msg)
                continue
            
            thread_id = data.get("thread_id")
            message = data.get("message")
            if "input_json" in data and not session_input_json:
                session_input_json = data["input_json"]
                print(f"Received input_json for session: {len(str(session_input_json))} bytes")

            # print("session_input_json:", data["input_json"])
            # Use thread-specific data if available, otherwise use session data or fallback to loaded data
            input_json_to_use = thread_data.get(thread_id, session_input_json or input_data)
            # print(f"Using input_json: {input_json_to_use} ")
            if not thread_id:
                error_msg = "Error: thread_id is required"
                print(error_msg)
                await websocket.send_text(error_msg)
                continue
                
            print(f"Processing message for thread_id: {thread_id}, message: {message}")
            
            config = {"configurable": {"user_id": "Test"}}
            graph_name = "retrieval_graph"

            if message == "." and not initial_message_handled:
                print("Starting initial report generation...")
                try:
                    await handle_initial_message_report(websocket, str(thread_id),input_json_to_use)
                    print("Initial report generation completed")
                    initial_message_handled = True
                except Exception as e:
                    error_msg = f"Error in initial report generation: {str(e)}"
                    print(error_msg)
                    await websocket.send_text(error_msg)
                continue

            # Get current thread state
            print("Retrieving thread state...")
            try:
                thread_state = await client.threads.get_state(thread_id)
                print(f"Thread state retrieved: {thread_state.keys()}")
                current_human_messages = thread_state['values'].get('human_messages', [])
                print(f"Current human messages count: {len(current_human_messages)}")
            except Exception as e:
                error_msg = f"Error retrieving thread state: {str(e)}"
                print(error_msg)
                await websocket.send_text(error_msg)
                continue
            
            # Create new human message
            new_human_message = {
                "content": message,
                "role": "human"
            }
            
            # Update messages
            updated_human_messages = current_human_messages + [new_human_message]
            print(f"Updated human messages count: {len(updated_human_messages)}")
            
            # Update thread state
            print("Updating thread state...")
            try:
                forked_config = await client.threads.update_state(
                    thread_id,
                    {"human_messages": updated_human_messages}
                )
                print(f"Thread state updated, checkpoint_id: {forked_config.get('checkpoint_id')}")
            except Exception as e:
                error_msg = f"Error updating thread state: {str(e)}"
                print(error_msg)
                await websocket.send_text(error_msg)
                continue
            
            # Continue the run
            print("Streaming run with updated state...")
            try:
                chunk_count = 0
                edited_found = False
                
                result2 = ""
                async for chunk in client.runs.stream(
                    thread_id,
                    graph_name,
                    input=None,
                    config=config,
                    checkpoint_id=forked_config['checkpoint_id'],
                    interrupt_before=["human_edit"],
                    stream_mode="messages-tuple"
                ):
                    
                    logger.info(f"========inside report after first message========")
                    logger.info(f"chunk : {chunk}")


                    
                    chunk_count += 1

                    if chunk.event == "messages":
                        result2 += "".join(data_item['content'] for data_item in chunk.data if 'content' in data_item)
                print(result2)
                for i in range(0, len(result2), 4):
                    chunk = result2[i:i+4]
                    await websocket.send_text(chunk)
                    await asyncio.sleep(0.008)


            except Exception as e:
                error_msg = f"Error in streaming run: {str(e)}"
                print(error_msg)
                await websocket.send_text(error_msg)
                continue
                    
    except WebSocketDisconnect:
        print("WebSocket disconnected by client")
    except Exception as e:
        error_msg = f"Unhandled error in WebSocket handler: {str(e)}"
        print(error_msg)
        try:
            await websocket.send_text(error_msg)
        except:
            pass
    finally:
        print("Closing WebSocket connection")
        await websocket.close()

#Research Agent
@app.post("/create_thread_research")
async def create_thread_research(input_json:Optional[Dict[str,Any]]=None):
    """Create a new thread for the LangGraph model"""
    url = 'http://climate_research_agent.railway.internal:8080'
    client = get_client(url=url)
    thread = await client.threads.create()
    return {"thread_id": thread["thread_id"]}


# async def websocket_endpoint_research(websocket: WebSocket):
#     """Simplified WebSocket endpoint for chat with LangGraph (no cleaning, just stream)."""
#     await websocket.accept()

#     url = 'https://research-agent-image-1032351832355.us-central1.run.app'
#     client = get_client(url=url)

#     try:
#         data = await websocket.receive_text()
#         data = json.loads(data)

#         thread_id = data.get("thread_id")
#         message = data.get("message")

#         if not thread_id:
#             await websocket.send_json({"error": "Thread ID is required"})
#             return

#         config = {'configurable': {"user_id": "123"}}
#         graph_name = "RetrievalGraph"

#         async for chunk in client.runs.stream(
#             thread_id,
#             graph_name,
#             input={"messages": [HumanMessage(content=message)]},
#             config=config,
#             stream_mode="messages-tuple"
#         ):
#             if chunk.event == "messages":
#                 for data_item in chunk.data:
#                     if 'content' in data_item:
#                         await websocket.send_text(data_item['content'])
#                         await asyncio.sleep(0.001)

#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print(f"Error: {e}")
#         await websocket.send_json({"error": str(e)})
#     finally:
#         await websocket.close()

@app.websocket("/chat_research")
async def websocket_endpoint_research(websocket: WebSocket):
    """WebSocket endpoint for interactive chat with the LangGraph model"""
    await websocket.accept()
    
    url = 'http://climate_research_agent.railway.internal:8080'
    client = get_client(url=url)
    

    try : 
        data = await websocket.receive_text()
        data = json.loads(data)
        logger.info(f"raw data recieved : {data}")

        thread_id = data.get("thread_id")
        message = data.get("message")
        
        if not thread_id:
            await websocket.send_json({"error": "Thread ID is required"})
            

        config = {'configurable': {"user_id": "123"}}
        graph_name = "retrieval_graph"

        result = ""
        async for chunk in client.runs.stream(thread_id, 
                                        graph_name, 
                                        input={"messages": [HumanMessage(content=message)]},
                                        config=config,
                                        stream_mode="messages-tuple"):
            logger.info(f"chunk : {chunk}")
            if chunk.event == "messages":
                result += "".join(data_item['content'] for data_item in chunk.data if 'content' in data_item)
        
        print(result)
        cleaned_message = re.sub(r'^\s*\{"logic":.?\}\s', '', result, flags=re.DOTALL)
        cleaned_message = re.sub(r'^\s*\{"queries":.?\}\s', '', cleaned_message, flags=re.DOTALL)
        cleaned_message = re.sub(r'^\s*\{"type":.?\}\s', '', cleaned_message, flags=re.DOTALL)
        cleaned_message = cleaned_message.split('@@@@@@')[-1].strip()
        print(cleaned_message)
        for i in range(0, len(cleaned_message), 3):
            chunk = cleaned_message[i:i+3]
            await websocket.send_text(chunk)
            await asyncio.sleep(0.001)


        thread_state = await client.threads.get_state(thread_id)


        if (len(thread_state["next"]) !=0 ):
            if ( thread_state["next"][0] == "More_Info"):
                print(f"Inside if statement")
                result = ""
                # Send a message to client requesting additional info
                # await websocket.send_json({"type": "input_request", "message": "Please provide additional information:"})
                await websocket.send_text("Please provide additional information...")
                
                # Wait for the user's response
                additional_info = await websocket.receive_text()
                additional_info = json.loads(additional_info)["message"]
                
                copied_thread = await client.threads.copy(thread_id)
                copied_thread_state = await client.threads.get_state(copied_thread['thread_id'])


                states_to_fork = await client.threads.get_history(thread_id)
                to_fork = states_to_fork[0]

                # Combine original message with additional info
                final_user_input = {"messages": HumanMessage(content=message + " " + additional_info, id=to_fork['values']['messages'][0]['id'])}
                
                forked_new_config = await client.threads.update_state(
                    thread_id,
                    final_user_input,
                    checkpoint_id=to_fork['checkpoint_id']
                    )
                
                print("Going to execute 2nd for loop...")
                async for chunk in client.runs.stream(thread_id, 
                                            graph_name, 
                                            input=None,
                                            config=config,
                                            checkpoint_id=forked_new_config['checkpoint_id'],
                                            stream_mode="messages-tuple"):
                    if chunk.event == "messages":
                        result += "".join(data_item['content'] for data_item in chunk.data if 'content' in data_item)
                        

                cleaned_message = re.sub(r'^\s*\{"logic":.?\}\s', '', result, flags=re.DOTALL)
                cleaned_message = re.sub(r'^\s*\{"queries":.?\}\s', '', cleaned_message, flags=re.DOTALL)
                cleaned_message = re.sub(r'^\s*\{"type":.?\}\s', '', cleaned_message, flags=re.DOTALL)
                print(cleaned_message)
                for i in range(0, len(cleaned_message), 3):
                    chunk = cleaned_message[i:i+3]
                    await websocket.send_text(chunk)
                    await asyncio.sleep(0.001)

                        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


class FormRequestModel(BaseModel):
    subregulation_id: str
    user_id: str

    @classmethod
    def as_form(cls, subregulation_id: str = Form(...), user_id: str = Form(...)):
        return cls(subregulation_id=subregulation_id, user_id=user_id)



def extract_formatted_document(text):
    # Look for the pattern that starts with "formatted_document': " followed by content
    pattern = r"formatted_document':\s*\"(.*?)(?:\"\s*,|\"\s*$|\]$)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Extract the content and handle escape characters
        content = match.group(1)
        # Replace escaped newlines with actual newlines if needed
        content = content.replace("\\n", "\n")
        return content
    else:
        # Try an alternative pattern if the first one fails
        alt_pattern = r"formatted_document':\s*(.*?)(?:,\s*\w+:|$|\])"
        alt_match = re.search(alt_pattern, text, re.DOTALL)
        if alt_match:
            content = alt_match.group(1).strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            content = content.replace("\\n", "\n")
            return content
        
        # If all else fails, try to find the section by looking for the markdown header
        header_pattern = r"# Greenhouse Gas Intensity of Corn Production(.*?)(?:\]$|\Z)"
        header_match = re.search(header_pattern, text, re.DOTALL)
        if header_match:
            return "# Greenhouse Gas Intensity of Corn Production" + header_match.group(1)
            
        return "Formatted document section not found"



def extract_edited_section(text):
    # Look for the pattern that starts with "edited': " or "edited: " followed by content
    pattern = r"edited'?:\s*\"?(.*?)(?:\"?\s*,|\"\s*$|\]$|$)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Extract the content and handle escape characters
        content = match.group(1).strip()
        # Remove surrounding quotes if present
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        # Replace escaped newlines with actual newlines if needed
        content = content.replace("\\n", "\n")
        return content
    else:
        # Try an alternative pattern if the first one fails
        alt_pattern = r"edited'?:\s*(.*?)(?:,\s*\w+:|$|\])"
        alt_match = re.search(alt_pattern, text, re.DOTALL)
        if alt_match:
            content = alt_match.group(1).strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            content = content.replace("\\n", "\n")
            return content
        
        # If all else fails, try to find the section by looking for the field directly
        field_pattern = r"edited'?:\s*(.*?)(?:human_messages|\Z)"
        field_match = re.search(field_pattern, text, re.DOTALL)
        if field_match:
            content = field_match.group(1).strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content
            
        return "Edited section not found"














# # in_memory_storage = in_memory_store
# # print("_______________________")
# # print(in_memory_store)
# # print(in_memory_store)
# # print("_________________")
# @app.post("/create_thread_training")
# async def create_thread_training(input_json:Optional[Dict[str,Any]]=None):
#     url_for_train_deployment = "training_agent.railway.internal"
#     # url_for_train_deployment = "http://127.0.0.1:2024"
#     client = get_client(url = url_for_train_deployment)
#     thread = await client.threads.create()
#     config ={"configurable":{"user_id":"Test"}}
#     graph_name = "training_agent"
#     print(in_memory_storage)
#     json_op={}
#     pattern = r"(structured_yield_state.*)"
#     async for chunk in client.runs.stream(
#         thread["thread_id"],
#         graph_name,
#         input={"messages": [HumanMessage(content="Extract all the information.")], "context" : in_memory_store },
#         config=config,
#         stream_mode="values"
#     ):
#         if chunk.event=="values":
#             json_op = chunk.data
#         # match = re.search(pattern, json_op, re.DOTALL)
#         # extracted_data = match.group(0)
        
#         #my_dict = list(json_op.items())
#         #json_op=json_op[2:]
#     print(type(json_op))
#     json_op = list(json_op.items())
#     json_op = json_op[2:]
#     return dict(json_op)

# @app.post("/generate_calculator_config")
# async def generate_calculator_config(
#     subregulation_id: str = Form(...),
#     user_id: str = Form(...),
#     pdfs: List[UploadFile] = File(...)
# ):
#     try:
#         # Process files and get markdown strings
#         all_markdown_content = []
#         filenames = []
#         for pdf in pdfs:
#             markdown_data = await process_file(pdf)
#             all_markdown_content.append(markdown_data)
#             filenames.append(pdf.filename)

#         # Concatenate all markdown content
#         combined_markdown = "\n\n---\n\n".join(all_markdown_content)

#         # Connect to training agent
#         url_for_train_deployment = "training_agent.railway.internal"
#         client = get_client(url=url_for_train_deployment)
#         thread = await client.threads.create()
#         config = {"configurable": {"user_id": "Test"}}
#         graph_name = "training_agent"

#         # Process combined markdown through training agent
#         json_op = {}
#         async for chunk in client.runs.stream(
#             thread["thread_id"],
#             graph_name,
#             input={
#                 "messages": [HumanMessage(content="Extract all the information.")],
#                 "context": combined_markdown
#             },
#             config=config,
#             stream_mode="values"
#         ):
#             if chunk.event == "values":
#                 json_op = chunk.data
        
#         # Format the output
#         json_op = list(json_op.items())
#         json_op = json_op[2:]  # Remove first two items as before
#         processed_data = dict(json_op)
#         response = {
#             "subregulation_id": subregulation_id,
#             "user_id": user_id,
#             "form_config": format_config_for_calculator(dict(json_op)),
#             "processed_files": {
#                 "filenames": filenames,
#                 # "extracted_data": dict(json_op)
#             }
#         }
#         print("Response generated successfully;")
#         return response

#     except Exception as e:
#         print("fallback done;")
#         return {
#             "subregulation_id": subregulation_id,
#             "user_id": user_id,
#             "form_config": format_config_for_calculator(form_config),
#         }
#         logger.error(f"Error in generate_calculator_config: {str(e)}")
        
#         raise HTTPException(status_code=500, detail=str(e))

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
from fastapi.routing import APIWebSocketRoute

print("Registered routes:")
for r in app.routes:
    print(type(r), r.path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)