import inspect
from datetime import datetime
import socket # Low-level networking interface (TCP/UDP, IP, hostname resolution).
import json
import uuid
from typing import Callable, List, Dict, Any, Optional, Callable, Union, get_args, get_origin
from dataclasses import is_dataclass, fields, MISSING
from pydantic import BaseModel # Data validation and settings management using Python type hints.
from rich.panel import Panel # Draw bordered panels.
from rich.prompt import Prompt # Interactive prompts.
from rich.console import Console # Flexible terminal output.
import inquirer # Interactive user prompts (checkboxes, lists, text input).
from rich.markdown import Markdown # Render markdown in the terminal.
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

# user_input = prompt("What's your name? ")
#print(f"Hello, {user_input}!")

# completer = WordCompleter(["python", "javascript", "rust"])
# user_input = prompt("Choose a language: ", completer=completer)



def debug_print_swarm(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")
    
def print_in_box(text: str, console: Optional[Console] = None, title: str = "", color: str = "white") -> None:
    """
    Print the text in a box.
    :param text: the text to print.
    :param console: the console to print the text.
    :param title: the title of the box.
    :param color: the border color.
    :return:
    """
    console = console or Console()

    # panel = Panel(text, title=title, border_style=color, expand=True, highlight=True)
    # console.print(panel)
    console.print('_'*20 + title + '_'*20, style=f"bold {color}")
    console.print(text, highlight=True, emoji=True)
    
def debug_print(debug: bool, *args: str, **kwargs: dict) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = "\n".join(map(str, args))
    color = kwargs.get("color", "white")
    title = kwargs.get("title", "")
    log_str = f"[{timestamp}]\n{message}"
    print_in_box(log_str, color=color, title=title)
    log_path = kwargs.get("log_path", None)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(log_str + '\n')



def ask_text(question: str, title: str = "User", console: Optional[Console] = None, default_answer: str = "") -> str:
    """
    Display a question in a panel and prompt the user for an answer.
    :param question: the question to display.
    :param title: the title of the panel.
    :param console: the console to use.
    :return: the user's answer.
    """
    console = console or Console()

    console.print(Panel(question, title=title, border_style="green"))
    answer = Prompt.ask(f"Type your answer here, press Enter to use default answer", default=default_answer)
    console.print(Panel(answer, title=title))
    return answer

def print_markdown(md_path: str, console: Optional[Console] = None):
    console = console or Console()
    with open(md_path, 'r') as f:
        md_content = f.read()
    console.print(Markdown(md_content))

def single_select_menu(options, message: str = ""): # Uses inquirer.List to create a single-selection interface
    questions = [
        inquirer.List(
            'choice', # 'choice': Key where the selection will be stored
            message=message,
            choices=options,
        ),
    ]
    answers = inquirer.prompt(questions)
    return answers['choice']


def get_user_confirmation(prompt: str) -> bool:
    user_input = prompt.strip().lower()
    if user_input in ['y', 'yes', 'true', 't']:
        return True
    elif user_input in ['n', 'no', 'false', 'f'] or user_input == '':
        return False
    else:
        print("Invalid input. Please enter 'y' for yes or 'n' for no.")

def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


# def function_to_json(func) -> dict:
#     """
#     Converts a Python function into a JSON-serializable dictionary
#     that describes the function's signature, including its name,
#     description, and parameters.

#     Args:
#         func: The function to be converted.

#     Returns:
#         A dictionary representing the function's signature in JSON format.
#     """
#     type_map = {
#         str: "string",
#         int: "integer",
#         float: "number",
#         bool: "boolean",
#         list: "array",
#         dict: "object",
#         type(None): "null",
#     }

#     try:
#         signature = inspect.signature(func) 
#     except ValueError as e:
#         raise ValueError(
#             f"Failed to get signature for function {func.__name__}: {str(e)}"
#         )

#     parameters = {}
#     for param in signature.parameters.values():
#         try:
#             param_type = type_map.get(param.annotation, "string")
#         except KeyError as e:
#             raise KeyError(
#                 f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
#             )
#         parameters[param.name] = {"type": param_type}

#     required = [
#         param.name
#         for param in signature.parameters.values()
#         if param.default == inspect._empty
#     ]

#     return {
#         "type": "function",
#         "function": {
#             "name": func.__name__,
#             "description": func.__doc__ or "",
#             "parameters": {
#                 "type": "object",
#                 "properties": parameters,
#                 "required": required,
#             },
#         },
#     }

# This function analyzes Python type annotations and returns structured type information in a standardized format
# primarily for converting Python types to JSON Schema-like representations.

# typing.get_origin(): Returns the base type of a type annotation, "Unwraps" special typing constructs to reveal the underlying type
# get_origin(List[int])         # Returns: list
# get_origin(Dict[str, int])    # Returns: dict
# get_origin(Union[int, str])   # Returns: typing.Union
# get_origin(Optional[float])   # Returns: typing.Union

# typing.get_args(): Returns the type parameters of a generic type; Shows what's inside the square brackets [] of a type annotation
# get_args(List[int])           # Returns: (int,)
# get_args(Dict[str, int])      # Returns: (str, int) 
# get_args(Union[int, str])     # Returns: (int, str)
# get_args(Optional[float])     # Returns: (float, type(None))

def get_type_info(annotation, base_type_map):
    # 处理基本类型
    if annotation in base_type_map:
        return {"type": base_type_map[annotation]}
    
    # 处理typing类型
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        
        # 处理List类型
        if origin is list or origin is List:
            item_type = args[0]
            return {
                "type": "array",
                "items": get_type_info(item_type, base_type_map)
            }
        
        # 处理Dict类型
        elif origin is dict or origin is Dict:
            key_type, value_type = args
            if key_type != str:
                raise ValueError("Dictionary keys must be strings")
            
            # 如果value_type是TypedDict或Pydantic模型
            if (hasattr(value_type, "__annotations__") or 
                (isinstance(value_type, type) and issubclass(value_type, BaseModel))):
                return get_type_info(value_type, base_type_map)
            
            # 普通Dict类型
            return {
                "type": "object",
                "additionalProperties": get_type_info(value_type, base_type_map)
            }
        
        # 处理Union类型
        # Processes all Union members except None (handles Optional[T] since Optional[T] = Union[T, None])
        # If only one non-None type remains, returns it directly (optimization)
        # e.g. Union[int, str] → {"oneOf": [{"type": "integer"}, {"type": "string"}]}
        # Optional[str] (≡ Union[str, None]) → {"type": "string"}
        
        elif origin is Union:
            types = [get_type_info(arg, base_type_map) for arg in args if arg != type(None)]
            if len(types) == 1:
                return types[0]
            return {"oneOf": types}
    
    # 处理Pydantic模型

    # Pydantic Model (Runtime-Validated Data Structures)
    # A runtime data validation library using Python type hints; Enforces types, converts data, and provides schemas

    
    if isinstance(annotation, type):
        try:
            if issubclass(annotation, BaseModel): # Attempts to verify if the class inherits from Pydantic's BaseModel
                schema = annotation.model_json_schema() # Uses Pydantic's built-in model_json_schema() to get the base schema
                # 提取主要schema部分
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                
                # 处理definitions
                definitions = schema.get("$defs", {})
                if definitions:
                    # 如果有引用的定义，直接展开它们
                    for prop_name, prop_schema in properties.items():
                        if "$ref" in prop_schema:
                            ref_name = prop_schema["$ref"].split("/")[-1]
                            if ref_name in definitions:
                                properties[prop_name] = definitions[ref_name]
                
                return {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
        except TypeError:
            pass
    
    # 处理dataclass
    # A dataclass is a decorator (@dataclass) that automatically generates common boilerplate code for classes, making them ideal for storing data. 
    # Introduced in Python 3.7 (via PEP 557), it significantly reduces verbose code while providing powerful features.
    '''
    @dataclass
    class Point:
        x: float
        y: float
    '''
    # This simple definition automatically gives you: __init__();__repr__();__eq__()
    # The field() function in Python's dataclasses module provides fine-grained control over how individual attributes behave in a dataclass. 
    # It's used to customize fields beyond what standard type annotations and default values can achieve.
    
    if is_dataclass(annotation):
        properties = {}
        required = []
        for field in fields(annotation):
            properties[field.name] = get_type_info(field.type, base_type_map)
            if field.default == field.default_factory == MISSING:
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }

    # 处理TypedDict
    # TypedDict (Python's Type-Annotated Dictionaries):A type annotation for dictionaries with fixed keys and value types
    # No runtime enforcement (pure type hint)

    
    if hasattr(annotation, "__annotations__"):
        properties = {}
        # Syntax: getattr(object, attribute_name, default): Tries to get attribute_name from object; Returns default if the attribute doesn't exist
        # __required_keys__: Specific to TypedDict (PEP 589); Contains a set of mandatory keys (non-optional fields)
        
        required = getattr(annotation, "__required_keys__", annotation.__annotations__.keys())
        
        for key, field_type in annotation.__annotations__.items():
            properties[key] = get_type_info(field_type, base_type_map)
        
        return {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False
        }

    # 默认返回string类型
    return {"type": "string"}


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        # list: "array",
        # dict: "object",
        type(None): "null",
    }
    # def get_type_info(annotation):
    #     if hasattr(annotation, "__origin__"):  # 处理typing类型
    #         origin = annotation.__origin__
    #         if origin is list:  # 处理List类型
    #             item_type = annotation.__args__[0]
    #             return {
    #                 "type": "array",
    #                 "items": {
    #                     "type": type_map.get(item_type, "string")
    #                 }
    #             }
    #         elif origin is dict:  # 处理Dict类型
    #             return {"type": "object"}
    #     return {"type": type_map.get(annotation, "string")}

    try:
        signature = inspect.signature(func) # inspect.signature(func): Extracts the function's parameter details (names, types, defaults).
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    # for param in signature.parameters.values():
    #     try:
    #         param_type = type_map.get(param.annotation, "string")
    #     except KeyError as e:
    #         raise KeyError(
    #             f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
    #         )
    #     parameters[param.name] = {"type": param_type}
    for param in signature.parameters.values(): # Returns an ordered dictionary view of a function's parameters.
        if param.name == "context_variables":
            continue
        try:
            param_info = get_type_info(param.annotation, type_map)
            if isinstance(param_info, dict) and "additionalProperties" in param_info:
                del param_info["additionalProperties"]
            parameters[param.name] = get_type_info(param.annotation, type_map)
        except KeyError as e:
            raise KeyError(f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}")

    

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # if not parameters:
    #     parameters["dummy"] = {
    #         "type": "string",
    #         "description": "Dummy parameter (not used). Added to satisfy non-empty schema requirements."
    #     }
    #     required = []

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def run_command_in_container_v1(command, stream_callback: Callable = None):
    # TCP parameters
    hostname = 'localhost'
    port = 12345  # TCP port mapped to the container
    buffer_size = 4096

    # Create TCP client
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((hostname, port))
        s.sendall(command.encode())
        full_response = b""
        while True:
            chunk = s.recv(buffer_size)
            if not chunk:
                break
            full_response += chunk
            if stream_callback:
                stream_callback(chunk)
            if len(chunk) < buffer_size:
                # If the received data is less than the buffer size, it may have been received
                break
        
        # Decode the complete response
        try:
            decoded_response = full_response.decode('utf-8')
            return json.loads(decoded_response)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response received: {decoded_response}")
            return {"status": -1, "result": "Response parsing error"}
            
# A Docker container is a lightweight, standalone, and executable software package that includes everything needed to run a piece of software:
# Code; Runtime (e.g., Python, Node.js, Java); System libraries; Environment variables; Configuration files
# Containers are created from Docker images, which are read-only templates defining the application environment.
# Isolation: Each container runs in its own isolated environment, separate from the host system and other containers.
# Portability: Containers can run consistently across different environments (development, testing, production).
# Lightweight & Fast: Unlike virtual machines (VMs), containers share the host OS kernel, making them much smaller and faster to start.

def run_command_in_container(command, stream_callback=None):
    """
    communicate with docker container and execute command, support stream output
    
    Args:
        command: the command to execute
        stream_callback: optional callback function, for handling stream output
                        the function signature should be callback(text: str)
    
    Returns:
        dict: the complete JSON result returned by the docker container
    """
    hostname = 'localhost'
    port = 12345
    buffer_size = 4096
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((hostname, port)) # Creates a TCP socket connection to localhost on port 12345
        s.sendall(command.encode()) # Sends the command to the container after encoding it to bytes
        
        partial_line = ""
        while True:
            chunk = s.recv(buffer_size) #Receives data in chunks of up to 4096 bytes
            if not chunk:
                break
                
            # add new received data to the unfinished data
            data = partial_line + chunk.decode('utf-8')
            lines = data.split('\n')
            
            # except the last line, process all complete lines
            for line in lines[:-1]:
                if line:
                    try:
                        response = json.loads(line) # parses a JSON-formatted string into a Python dictionary (or a list, depending on the JSON structure).
                        if response['type'] == 'chunk':
                            # process stream output
                            if stream_callback:
                                stream_callback(response['data']) # Progressively sending chunks of output data (as they arrive) to a user-defined function, instead of waiting for all the data at once.
                        elif response['type'] == 'final':
                            # return the final result
                            return {
                                'status': response['status'],
                                'result': response['result']
                            }
                    except json.JSONDecodeError:
                        print(f"Invalid JSON: {line}")
            
            # save the possibly unfinished last line
            partial_line = lines[-1]
            
    # if the loop ends normally without receiving a final response
    return {
        'status': -1,
        'result': 'Connection closed without final response'
    }
        

def make_tool_message(tools: Callable, args: dict, tool_content: str) -> List[Dict]:
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": tools.__name__,
                "arguments": json.dumps(args)
            }, 
            "id": str(uuid.uuid4()).replace('-', '')[:9]
        }
    ]
    return [
        {'role': 'assistant', 'tool_calls': tool_calls},
        {'role': 'tool', 'content': tool_content, 'name': tools.__name__, 'tool_call_id': tool_calls[0]['id']}
    ]
def make_message(role: str, content: str):
    return [
        {'role': role, 'content': content}
    ]




class UserCompleter(Completer):

    def __init__(self, users: List[str]):
        super().__init__()
        self.users = users
    def get_completions(self, document, complete_event):
        word = document.get_word_before_cursor()
        
        if word.startswith('@'):
            prefix = word[1:]  # 去掉@
            for user in self.users:
                if user.startswith(prefix):
                    yield Completion(
                        user,
                        start_position=-len(prefix),
                        style='fg:blue bold'  # 蓝色加粗
                    )
def pretty_print_messages(message, **kwargs) -> None:
    # for message in messages:
    if message["role"] != "assistant" and message["role"] != "tool":
        return
    console = Console()
    if message["role"] == "tool":
        console.print("[bold blue]tool execution:[/bold blue]", end=" ")
        console.print(f"[bold purple]{message['name']}[/bold purple], result: {message['content']}")
        log_path = kwargs.get("log_path", None)
        if log_path:
            with open(log_path, 'a') as file:
                file.write(f"tool execution: {message['name']}, result: {message['content']}\n")
        return
                
    # print agent name in blue
    console.print(f"[bold blue]{message['sender']}[/bold blue]:", end=" ")

    # print response, if any
    if message["content"]:
        console.print(message["content"], highlight=True, emoji=True)

    # print tool calls in purple, if any
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) > 1:
        console.print()
    for tool_call in tool_calls:
        f = tool_call["function"]
        name, args = f["name"], f["arguments"]
        arg_str = json.dumps(json.loads(args)).replace(":", "=")
        console.print(f"[bold purple]{name}[/bold purple]({arg_str[1:-1]})")
    log_path = kwargs.get("log_path", None)
    if log_path:
        with open(log_path, 'a') as file:
            file.write(f"{message['sender']}: {message['content']}\n")
            for tool_call in tool_calls:
                f = tool_call["function"]
                name, args = f["name"], f["arguments"]
                arg_str = json.dumps(json.loads(args)).replace(":", "=")
                file.write(f"{name}({arg_str[1:-1]})\n")

