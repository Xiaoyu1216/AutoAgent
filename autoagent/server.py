# FastAPI is a modern, high-performance Python web framework for building APIs (Application Programming Interfaces). It is designed for speed, ease of use, and automatic documentation. 
# 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from autoagent.registry import registry
from autoagent import MetaChain
from autoagent.types import Agent, Response
import importlib
import inspect

# A decorator is a powerful Python feature that modifies or enhances functions or classes without permanently changing their source code. 
# Think of it as "wrapping" your existing code to add new capabilities.
# GET	Retrieve data	Fetching a webpage, loading user profile
# POST	Send/submit data	Login forms, file uploads, API commands

# A context manager is a Python construct that: Sets up resources when entering a block of code (e.g., opening a file). 
# Cleans up resources when exiting the block (e.g., closing the file).

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_agent_endpoints(app)
    yield
    

app = FastAPI(title="MetaChain API", lifespan=lifespan)

class ToolRequest(BaseModel):
    args: Dict[str, Any]

class AgentRequest(BaseModel):
    model: str
    query: str
    context_variables: Optional[Dict[str, Any]] = {}

class Message(BaseModel):
    role: str
    content: str

class AgentResponse(BaseModel):
    result: str
    messages: List
    agent_name: str

@app.on_event("startup")
def create_tool_endpoints():
    for tool_name, tool_func in registry.tools.items():
        
        async def create_tool_endpoint(request: ToolRequest, func=tool_func):
            try:
                
                sig = inspect.signature(func)
                required_params = {
                    name for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                }
                
                
                if not all(param in request.args for param in required_params):
                    missing = required_params - request.args.keys()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required parameters: {missing}"
                    )
                
                result = func(**request.args)
                return {"status": "success", "result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        endpoint = create_tool_endpoint
        endpoint.__name__ = f"tool_{tool_name}"
        app.post(f"/tools/{tool_name}")(endpoint)
        
# 重写agent endpoints创建逻辑
@app.on_event("startup")
def create_agent_endpoints():
    for agent_name, agent_func in registry.agents.items():
        async def create_agent_endpoint(
            request: AgentRequest, 
            func=agent_func
        ) -> AgentResponse:
            try:
                # 创建agent实例
                agent = func(model=request.model)
                
                # 创建MetaChain实例
                mc = MetaChain()
                
                # 构建messages
                messages = [
                    {"role": "user", "content": request.query}
                ]
                
                # 运行agent
                response = mc.run(
                    agent=agent,
                    messages=messages,
                    context_storage=request.context_variables,
                    debug=True
                )
                
                return AgentResponse(
                    result=response.messages[-1]['content'],
                    messages=response.messages,
                    agent_name=agent.name
                )
                
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent execution failed: {str(e)}"
                )
        
        endpoint = create_agent_endpoint
        endpoint.__name__ = f"agent_{agent_name}"
        app.post(f"/agents/{agent_name}/run")(endpoint)

@app.get("/agents")
async def list_agents():
    return {
        name: {
            "docstring": info.docstring,
            "args": info.args,
            "file_path": info.file_path
        }
        for name, info in registry.agents_info.items()
    }

@app.get("/agents/{agent_name}")
async def get_agent_info(agent_name: str):
    if agent_name not in registry.agents_info:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_name} not found"
        )
    
    info = registry.agents_info[agent_name]
    return {
        "name": agent_name,
        "docstring": info.docstring,
        "args": info.args,
        "file_path": info.file_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
