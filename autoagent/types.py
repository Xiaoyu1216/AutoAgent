# LiteLLM is an open-source Python library designed to standardize interactions with multiple large language model (LLM) APIs (e.g., OpenAI, Anthropic, Cohere, HuggingFace, etc.). 

from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message
from typing import List, Callable, Union, Optional, Tuple, Dict

# Third-party imports
from pydantic import BaseModel

# A type alias in Python (defined using NewType or simple assignment) creates a new name for an existing type to improve code clarity, maintainability, and type safety. 
# In Python's type system, Callable is a type hint that indicates something is a function or callable object (like a class with __call__). It describes:
# Input parameters (arguments the function accepts); Return type (what the function returns);
# Callable[[Arg1Type, Arg2Type, ...], ReturnType]


AgentFunction = Callable[[], Union[str, "Agent", dict]]

# []: The function takes no arguments (empty list). 
# Union[str, "Agent", dict]: The function returns one of three types: str (a string), "Agent" (an instance of the Agent class), dict (a dictionary).

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    # functions: The name of the class attribute (field).
    # List[AgentFunction]: Type hint specifying this is a list where each item is an AgentFunction.
    # = []: Default value (empty list).
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = False
    examples: Union[List[Tuple[dict, str]], Callable[[], str]] = []
    handle_mm_func: Callable[[], str] = None
    agent_teams: Dict[str, Callable] = {}


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
    image: Optional[str] = None # base64 encoded image
