"""
============================================================================
LANGGRAPH COMPLETE TUTORIAL
============================================================================

A comprehensive guide to building stateful, multi-actor agent workflows with LangGraph.

LangGraph is a low-level orchestration framework for building, managing, and deploying
long-running, stateful agents. It's built on the Pregel model (inspired by Google's Pregel).

Key Concepts:
- State: Shared data structure representing the current snapshot
- Nodes: Functions that perform work and update state
- Edges: Functions that determine which node to execute next
- Graphs: Composed of nodes and edges to create workflows

This tutorial covers everything from basics to advanced patterns.
"""

# ============================================================================
# PART 1: ESSENTIAL IMPORTS FOR LANGGRAPH
# ============================================================================

# Core LangGraph components
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain components (for models and tools)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

# Type hints
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from typing_extensions import TypedDict as ExtTypedDict

# Utilities
import os
from operator import add


# ============================================================================
# PART 2: UNDERSTANDING LANGGRAPH - THE BIG PICTURE
# ============================================================================

"""
LANGGRAPH WORKFLOW:

1. DEFINE STATE: Create a TypedDict that represents your application's state
   └─> State can contain messages, data, flags, etc.

2. CREATE NODES: Define functions that:
   - Receive the current state
   - Perform computation (LLM calls, tool calls, etc.)
   - Return updated state

3. CREATE EDGES: Define how nodes connect:
   - Fixed edges: Always go to the same next node
   - Conditional edges: Choose next node based on state

4. BUILD GRAPH: Compose nodes and edges into a graph
   └─> StateGraph API makes this easy

5. COMPILE: Compile the graph into an executable workflow
   └─> Can add checkpointing, streaming, etc.

6. INVOKE: Run the graph with initial state

WHY LANGGRAPH?
- Durable execution: Agents persist through failures
- Human-in-the-loop: Inspect and modify state at any point
- Comprehensive memory: Short-term and long-term memory
- Streaming: Real-time updates
- Production-ready: Built for long-running, stateful workflows

LANGGRAPH VS LANGCHAIN:
- LangChain: High-level abstractions, quick to build agents
- LangGraph: Low-level orchestration, full control over workflows
- LangChain's create_agent is built on LangGraph!
"""


# ============================================================================
# PART 3: BASIC CONCEPTS - STATE
# ============================================================================

def example_1_define_state():
    """
    Define a state schema using TypedDict.
    
    State is the shared data structure that flows through your graph.
    Each node can read from and write to the state.
    """
    print("\n--- Example 1: Defining State ---")
    
    # Simple state with a counter
    class CounterState(TypedDict):
        count: int
        history: List[str]
    
    # State with messages (common for chat agents)
    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]  # Messages are appended
        counter: int  # Simple value
    
    # State with multiple data types
    class ComplexState(TypedDict):
        messages: Annotated[List[Any], add_messages]
        user_data: Dict[str, Any]
        flags: Dict[str, bool]
        step_count: int
    
    print("✅ State schemas defined")
    print("   - CounterState: Simple counter with history")
    print("   - AgentState: Messages + counter")
    print("   - ComplexState: Multiple data types")


# ============================================================================
# PART 4: BASIC CONCEPTS - NODES
# ============================================================================

def example_2_create_nodes():
    """
    Create node functions.
    
    Nodes are functions that:
    - Take state as input
    - Perform work (LLM calls, calculations, etc.)
    - Return updated state
    """
    print("\n--- Example 2: Creating Nodes ---")
    
    # Simple state
    class SimpleState(TypedDict):
        value: int
        log: List[str]
    
    # Node 1: Increment counter
    def increment_node(state: SimpleState) -> SimpleState:
        """Increment the value and log it."""
        new_value = state["value"] + 1
        new_log = state["log"] + [f"Incremented to {new_value}"]
        return {"value": new_value, "log": new_log}
    
    # Node 2: Double the value
    def double_node(state: SimpleState) -> SimpleState:
        """Double the value and log it."""
        new_value = state["value"] * 2
        new_log = state["log"] + [f"Doubled to {new_value}"]
        return {"value": new_value, "log": new_log}
    
    print("✅ Nodes created:")
    print("   - increment_node: Adds 1 to value")
    print("   - double_node: Multiplies value by 2")
    
    return increment_node, double_node


# ============================================================================
# PART 5: BASIC CONCEPTS - EDGES
# ============================================================================

def example_3_create_edges():
    """
    Create edges to connect nodes.
    
    Edges determine the flow of execution:
    - Fixed edges: Always go to the same node
    - Conditional edges: Choose next node based on state
    """
    print("\n--- Example 3: Creating Edges ---")
    
    class SimpleState(TypedDict):
        value: int
        step: str
    
    # Conditional edge function
    def should_continue(state: SimpleState) -> Literal["increment", "double", END]:
        """Decide which node to execute next."""
        value = state["value"]
        
        if value < 5:
            return "increment"
        elif value < 20:
            return "double"
        else:
            return END  # Stop execution
    
    print("✅ Conditional edge created:")
    print("   - If value < 5: go to 'increment'")
    print("   - If value < 20: go to 'double'")
    print("   - Otherwise: END")
    
    return should_continue


# ============================================================================
# PART 6: BUILDING YOUR FIRST GRAPH
# ============================================================================

def example_4_build_simple_graph():
    """
    Build a simple graph with nodes and edges.
    """
    print("\n--- Example 4: Building a Simple Graph ---")
    
    # Define state
    class CounterState(TypedDict):
        count: int
        history: List[str]
    
    # Define nodes
    def increment(state: CounterState) -> CounterState:
        new_count = state["count"] + 1
        return {
            "count": new_count,
            "history": state["history"] + [f"Count is now {new_count}"]
        }
    
    def decrement(state: CounterState) -> CounterState:
        new_count = state["count"] - 1
        return {
            "count": new_count,
            "history": state["history"] + [f"Count is now {new_count}"]
        }
    
    # Conditional edge
    def route(state: CounterState) -> Literal["increment", "decrement", END]:
        if state["count"] < 5:
            return "increment"
        elif state["count"] > 0:
            return "decrement"
        else:
            return END
    
    # Build graph
    workflow = StateGraph(CounterState)
    workflow.add_node("increment", increment)
    workflow.add_node("decrement", decrement)
    
    # Set entry point
    workflow.add_edge(START, "increment")
    
    # Add conditional edge
    workflow.add_conditional_edges("increment", route)
    workflow.add_conditional_edges("decrement", route)
    
    # Compile
    graph = workflow.compile()
    
    print("✅ Graph built and compiled!")
    print("   Graph structure:")
    print("   START -> increment -> [conditional] -> increment/decrement/END")
    print("   decrement -> [conditional] -> increment/decrement/END")
    
    # Run the graph
    initial_state = {"count": 0, "history": []}
    result = graph.invoke(initial_state)
    
    print(f"\n   Final count: {result['count']}")
    print(f"   History: {result['history']}")
    
    return graph


# ============================================================================
# PART 7: BUILDING A CHAT AGENT WITH LANGGRAPH
# ============================================================================

def example_5_chat_agent(api_key: str):
    """
    Build a simple chat agent using LangGraph.
    
    This demonstrates:
    - State with messages
    - LLM integration
    - Tool calling
    - Conditional routing
    """
    print("\n--- Example 5: Building a Chat Agent ---")
    
    # Define tools
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny with 25°C."
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            result = eval(expression)
            return f"The result is {result}"
        except:
            return "Invalid expression"
    
    tools = [get_weather, calculate]
    
    # Define state
    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Node: Call LLM
    def call_model(state: AgentState):
        """Call the LLM with current messages."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Conditional edge: Should we continue?
    def should_continue(state: AgentState) -> Literal["tools", END]:
        """Decide if we need to call tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If LLM made tool calls, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, we're done
        return END
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.add_edge(START, "agent")
    
    # Add conditional edge from agent
    workflow.add_conditional_edges("agent", should_continue)
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile
    graph = workflow.compile()
    
    print("✅ Chat agent graph built!")
    print("   Graph structure:")
    print("   START -> agent -> [if tool_calls] -> tools -> agent")
    print("   agent -> [if no tool_calls] -> END")
    
    # Test the agent
    initial_state = {
        "messages": [HumanMessage(content="What's the weather in London?")]
    }
    
    print("\n   Testing agent...")
    result = graph.invoke(initial_state)
    print(f"   Final message: {result['messages'][-1].content}")
    
    return graph


# ============================================================================
# PART 8: ADVANCED PATTERNS - STREAMING
# ============================================================================

def example_6_streaming(api_key: str):
    """
    Add streaming to your graph for real-time updates.
    """
    print("\n--- Example 6: Streaming ---")
    
    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    
    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    
    graph = workflow.compile()
    
    print("✅ Graph with streaming support!")
    print("   Use graph.stream() to get real-time updates:")
    print("   for chunk in graph.stream(initial_state):")
    print("       print(chunk)")
    
    return graph


# ============================================================================
# PART 9: ADVANCED PATTERNS - CHECKPOINTING
# ============================================================================

def example_7_checkpointing(api_key: str):
    """
    Add checkpointing for durable execution.
    
    Checkpointing allows:
    - Resume from failures
    - Human-in-the-loop
    - State persistence
    """
    print("\n--- Example 7: Checkpointing ---")
    
    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    
    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    
    # Add checkpointing
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    print("✅ Graph with checkpointing!")
    print("   - State is saved at each step")
    print("   - Can resume from any checkpoint")
    print("   - Supports human-in-the-loop")
    
    # Use with thread_id for conversation persistence
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    return graph, config


# ============================================================================
# PART 10: ADVANCED PATTERNS - HUMAN-IN-THE-LOOP
# ============================================================================

def example_8_human_in_the_loop(api_key: str):
    """
    Add human approval before tool execution.
    """
    print("\n--- Example 8: Human-in-the-Loop ---")
    
    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email."""
        return f"Email sent to {to} with subject '{subject}'"
    
    tools = [send_email]
    
    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]
        approved_tools: List[Dict[str, Any]]
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def approve_tools(state: AgentState):
        """Approve tools (in real app, this would wait for human input)."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # In real app, show tool calls to human and wait for approval
            print("   [Human approval needed for tool calls]")
            # For demo, auto-approve
            return {"approved_tools": last_message.tool_calls}
        
        return {"approved_tools": []}
    
    def should_continue(state: AgentState) -> Literal["approve", "tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if not state.get("approved_tools"):
                return "approve"  # Need approval first
            return "tools"
        
        return END
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("approve", approve_tools)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("approve", "tools")
    workflow.add_edge("tools", "agent")
    
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    print("✅ Graph with human-in-the-loop!")
    print("   Flow: agent -> approve -> tools -> agent")
    print("   Human can approve/reject tool calls before execution")
    
    return graph


# ============================================================================
# PART 11: BUILDING A RAG AGENT WITH LANGGRAPH
# ============================================================================

def example_9_rag_agent(api_key: str):
    """
    Build a RAG agent that decides when to retrieve documents.
    """
    print("\n--- Example 9: RAG Agent ---")
    
    # This is a simplified version - in practice, you'd have a real vector store
    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]
        retrieved_docs: List[str]
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    
    @tool
    def retrieve_documents(query: str) -> str:
        """Retrieve relevant documents for a query."""
        # Simulated retrieval
        docs = [
            "Python is a programming language.",
            "LangGraph is a framework for building agents.",
            "RAG stands for Retrieval Augmented Generation."
        ]
        return "\n".join(docs)
    
    tools = [retrieve_documents]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def call_model(state: AgentState):
        messages = state["messages"]
        
        # Add retrieved docs to context if available
        if state.get("retrieved_docs"):
            context = "\n\n".join(state["retrieved_docs"])
            system_msg = SystemMessage(
                content=f"Use this context to answer questions:\n\n{context}"
            )
            messages = [system_msg] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def store_retrieved_docs(state: AgentState):
        """Store retrieved documents in state."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Extract retrieved docs from tool results
            # In practice, this would come from the tool execution
            return {"retrieved_docs": ["Document 1", "Document 2"]}
        
        return {}
    
    def should_continue(state: AgentState) -> Literal["retrieve", "tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_name = last_message.tool_calls[0]["name"]
            if tool_name == "retrieve_documents":
                return "retrieve"
            return "tools"
        
        return END
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("retrieve", store_retrieved_docs)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("retrieve", "agent")  # After retrieval, go back to agent
    workflow.add_edge("tools", "agent")
    
    graph = workflow.compile()
    
    print("✅ RAG agent graph built!")
    print("   Flow: agent -> [if retrieve needed] -> retrieve -> agent")
    print("   Agent decides when to retrieve documents")
    
    return graph


# ============================================================================
# PART 12: COMMAND API - COMBINING CONTROL FLOW AND STATE UPDATES
# ============================================================================

def example_10_command_api():
    """
    Use Command API to combine control flow and state updates.
    
    Command allows you to:
    - Update state
    - Decide next node
    - All in one function
    """
    print("\n--- Example 10: Command API ---")
    
    from langgraph.graph import Command
    
    class State(TypedDict):
        value: int
        path: List[str]
    
    def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
        """Node A updates state and decides next node."""
        new_value = state["value"] + 1
        
        # Randomly choose next node (in practice, use logic)
        import random
        next_node = "node_b" if random.random() > 0.5 else "node_c"
        
        return Command(
            update={"value": new_value, "path": state["path"] + ["A"]},
            goto=next_node
        )
    
    def node_b(state: State) -> State:
        return {"value": state["value"] * 2, "path": state["path"] + ["B"]}
    
    def node_c(state: State) -> State:
        return {"value": state["value"] - 1, "path": state["path"] + ["C"]}
    
    workflow = StateGraph(State)
    workflow.add_node("node_a", node_a)
    workflow.add_node("node_b", node_b)
    workflow.add_node("node_c", node_c)
    
    workflow.add_edge(START, "node_a")
    # No edges needed! Command handles routing
    
    graph = workflow.compile()
    
    print("✅ Graph using Command API!")
    print("   - node_a updates state AND decides next node")
    print("   - No explicit edges needed from node_a")
    
    return graph


# ============================================================================
# PART 13: PARALLEL EXECUTION
# ============================================================================

def example_11_parallel_execution():
    """
    Execute multiple nodes in parallel.
    """
    print("\n--- Example 11: Parallel Execution ---")
    
    class State(TypedDict):
        results: Dict[str, int]
    
    def process_a(state: State) -> State:
        return {"results": {**state["results"], "a": 10}}
    
    def process_b(state: State) -> State:
        return {"results": {**state["results"], "b": 20}}
    
    def process_c(state: State) -> State:
        return {"results": {**state["results"], "c": 30}}
    
    workflow = StateGraph(State)
    workflow.add_node("a", process_a)
    workflow.add_node("b", process_b)
    workflow.add_node("c", process_c)
    
    # Fan out: START -> all three nodes in parallel
    workflow.add_edge(START, "a")
    workflow.add_edge(START, "b")
    workflow.add_edge(START, "c")
    
    # Fan in: All nodes -> END
    workflow.add_edge("a", END)
    workflow.add_edge("b", END)
    workflow.add_edge("c", END)
    
    graph = workflow.compile()
    
    print("✅ Graph with parallel execution!")
    print("   - Nodes a, b, c execute in parallel")
    print("   - Results are merged in state")
    
    return graph


# ============================================================================
# PART 14: LANGGRAPH BEST PRACTICES
# ============================================================================

"""
LANGGRAPH BEST PRACTICES:

1. STATE DESIGN:
   - Keep state minimal and focused
   - Use Annotated for reducers (add_messages, etc.)
   - Avoid deeply nested structures
   - Use TypedDict for type safety

2. NODE DESIGN:
   - Make nodes idempotent when possible
   - Keep nodes focused on single responsibilities
   - Handle errors gracefully
   - Return partial state updates

3. EDGE DESIGN:
   - Use conditional edges for dynamic routing
   - Keep routing logic simple
   - Document edge conditions clearly
   - Use END explicitly when done

4. CHECKPOINTING:
   - Always use checkpointing for production
   - Use thread_id for conversation persistence
   - Consider checkpoint size for large states
   - Test resume from checkpoints

5. STREAMING:
   - Stream for better UX
   - Handle streaming errors
   - Consider what to stream (state, messages, etc.)

6. ERROR HANDLING:
   - Wrap LLM calls in try-except
   - Handle tool failures gracefully
   - Provide fallback paths
   - Log errors for debugging

7. TESTING:
   - Test each node independently
   - Test edge conditions
   - Test with various state configurations
   - Test checkpointing and resumption

8. PERFORMANCE:
   - Use parallel execution when possible
   - Cache expensive operations
   - Optimize state size
   - Monitor execution time
"""


# ============================================================================
# PART 15: LANGGRAPH VS LANGCHAIN AGENTS
# ============================================================================

"""
WHEN TO USE LANGGRAPH VS LANGCHAIN:

Use LangChain's create_agent() when:
- You want to quickly build standard agent patterns
- ReAct or similar patterns are sufficient
- You prefer high-level abstractions
- You don't need custom orchestration

Use LangGraph directly when:
- You need custom control flow
- You want fine-grained control over state
- You need complex conditional logic
- You want to implement custom patterns
- You need human-in-the-loop
- You need durable execution with checkpointing
- You're building multi-agent systems

Remember: LangChain's create_agent() is built on LangGraph!
You can start with LangChain and drop down to LangGraph when needed.
"""


# ============================================================================
# PART 16: COMPLETE EXAMPLE - CUSTOM AGENT
# ============================================================================

def example_12_complete_agent(api_key: str):
    """
    Complete example: A custom agent with all features.
    """
    print("\n--- Example 12: Complete Custom Agent ---")
    
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        return f"Search results for: {query}"
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            return str(eval(expression))
        except:
            return "Error"
    
    tools = [search_web, calculate]
    
    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]
        tool_calls_count: int
        step: str
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "step": "model_called"
        }
    
    def execute_tools(state: AgentState):
        tool_calls = state.get("tool_calls_count", 0) + 1
        return {
            "tool_calls_count": tool_calls,
            "step": "tools_executed"
        }
    
    def should_continue(state: AgentState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return END
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("track", execute_tools)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "track")
    workflow.add_edge("track", "agent")
    
    # Add checkpointing
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    print("✅ Complete agent built!")
    print("   Features:")
    print("   - LLM integration")
    print("   - Tool calling")
    print("   - State tracking")
    print("   - Checkpointing")
    print("   - Conditional routing")
    
    return graph


# ============================================================================
# 🧪 TEST YOUR KNOWLEDGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 LANGGRAPH COMPLETE TUTORIAL")
    print("="*70)
    print("\nThis tutorial covers:")
    print("  ✓ State, Nodes, and Edges")
    print("  ✓ Building graphs")
    print("  ✓ Chat agents")
    print("  ✓ RAG agents")
    print("  ✓ Streaming and checkpointing")
    print("  ✓ Human-in-the-loop")
    print("  ✓ Advanced patterns")
    print("\nUncomment examples below to try them!")
    print("(Make sure to set your OPENAI_API_KEY first)\n")
    
    # Example usage (uncomment to run):
    # api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")
    # 
    # example_1_define_state()
    # example_2_create_nodes()
    # example_3_create_edges()
    # example_4_build_simple_graph()
    # example_5_chat_agent(api_key)
    # example_6_streaming(api_key)
    # example_7_checkpointing(api_key)
    # example_8_human_in_the_loop(api_key)
    # example_9_rag_agent(api_key)
    # example_10_command_api()
    # example_11_parallel_execution()
    # example_12_complete_agent(api_key)
    
    print("\n✅ LangGraph Tutorial complete!")
    print("\nNext steps:")
    print("  1. Build your own custom agent")
    print("  2. Experiment with different state schemas")
    print("  3. Try checkpointing and streaming")
    print("  4. Deploy to production with LangSmith")

