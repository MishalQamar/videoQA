"""
🎓 LangChain + OpenAI Complete Tutorial
========================================

This tutorial teaches you everything you need to know about LangChain with OpenAI.
Follow along step by step!
"""

# ============================================================================
# 📦 PART 1: ESSENTIAL IMPORTS (What you need to know)
# ============================================================================

# 1. Chat Model - Connect to OpenAI
from langchain_openai import ChatOpenAI
# This is how you connect LangChain to OpenAI's chat models (GPT-4, GPT-3.5, etc.)

# 2. Prompts - Create structured prompts
from langchain_core.prompts import ChatPromptTemplate
# This helps you create reusable prompt templates with variables

# 3. Output Parsers - Format model responses
from langchain_core.output_parsers import StrOutputParser
# Converts model responses to strings (or other formats)

# 4. Documents - Work with text data
from langchain_core.documents import Document
# Represents a piece of text with metadata

# 5. Document Loaders - Load data from various sources
from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader
# Load documents from YouTube, PDFs, websites, etc.

# ============================================================================
# 🎯 PART 2: CORE CONCEPTS
# ============================================================================

"""
LangChain has 3 main building blocks:

1. PROMPTS → Instructions for the AI
2. MODELS → The AI (OpenAI, Anthropic, etc.)
3. OUTPUT PARSERS → Format the response

You chain them together using LCEL (LangChain Expression Language):
    prompt | model | output_parser
"""

# ============================================================================
# 📝 PART 3: BASIC EXAMPLE - Simple Chat
# ============================================================================

def example_1_simple_chat():
    """Example 1: Simple one-shot chat with OpenAI"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Chat")
    print("="*60)
    
    # Step 1: Create the model (connect to OpenAI)
    model = ChatOpenAI(
        model="gpt-4o-mini",  # Choose your model
        api_key="your-api-key-here",  # Your OpenAI API key
        temperature=0.7,  # Creativity (0=deterministic, 1=creative)
    )
    
    # Step 2: Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}")
    ])
    
    # Step 3: Create the chain (prompt → model → parser)
    chain = prompt | model | StrOutputParser()
    
    # Step 4: Use it!
    response = chain.invoke({"question": "What is Python?"})
    print(f"Response: {response}")
    
    return response


# ============================================================================
# 📊 PART 4: SUMMARIZATION EXAMPLE
# ============================================================================

def example_2_summarization():
    """Example 2: Summarize text using LangChain chain"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Text Summarization")
    print("="*60)
    
    # Long text to summarize
    long_text = """
    Python is a high-level programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991. Python supports multiple
    programming paradigms including procedural, object-oriented, and functional programming.
    It has a large standard library and is widely used in web development, data science,
    artificial intelligence, and automation. Python's syntax allows programmers to express
    concepts in fewer lines of code than would be possible in languages such as C++ or Java.
    """
    
    # Step 1: Create model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="your-api-key-here",
        temperature=0,  # Lower temperature for more consistent summaries
    )
    
    # Step 2: Create summarization prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at summarizing text. Provide concise summaries."),
        ("user", "Summarize the following text in 2-3 sentences:\n\n{text}")
    ])
    
    # Step 3: Create chain
    chain = prompt | model | StrOutputParser()
    
    # Step 4: Summarize
    summary = chain.invoke({"text": long_text})
    print(f"Summary: {summary}")
    
    return summary


# ============================================================================
# 💬 PART 5: CHATBOT WITH MEMORY
# ============================================================================

def example_3_chatbot():
    """Example 3: Chatbot that remembers conversation"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Chatbot with Conversation Memory")
    print("="*60)
    
    from langchain_core.messages import HumanMessage, AIMessage
    
    # Step 1: Create model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="your-api-key-here",
    )
    
    # Step 2: Create prompt with conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly chatbot. Remember the conversation."),
        ("placeholder", "{messages}"),  # This is where conversation history goes
    ])
    
    # Step 3: Create chain
    chain = prompt | model
    
    # Step 4: Simulate a conversation
    conversation = []
    
    # First message
    conversation.append(HumanMessage(content="Hi! My name is Alice."))
    response1 = chain.invoke({"messages": conversation})
    conversation.append(response1)
    print(f"Bot: {response1.content}")
    
    # Second message (bot should remember the name)
    conversation.append(HumanMessage(content="What's my name?"))
    response2 = chain.invoke({"messages": conversation})
    conversation.append(response2)
    print(f"Bot: {response2.content}")
    
    return conversation


# ============================================================================
# 📄 PART 6: DOCUMENT PROCESSING
# ============================================================================

def example_4_document_qa():
    """Example 4: Question-Answering over documents"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Document Question-Answering")
    print("="*60)
    
    # Step 1: Create documents (in real app, load from file/URL)
    documents = [
        Document(
            page_content="Python is a programming language created in 1991.",
            metadata={"source": "intro.txt"}
        ),
        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "langchain.txt"}
        ),
    ]
    
    # Step 2: Combine documents into context
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Step 3: Create Q&A prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer questions based on the following context:\n\n{context}"),
        ("user", "Question: {question}")
    ])
    
    # Step 4: Create model and chain
    model = ChatOpenAI(model="gpt-4o-mini", api_key="your-api-key-here")
    chain = prompt | model | StrOutputParser()
    
    # Step 5: Ask a question
    question = "When was Python created?"
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    return answer


# ============================================================================
# 🔗 PART 7: ADVANCED - MULTI-STEP CHAIN
# ============================================================================

def example_5_multi_step():
    """Example 5: Multi-step processing chain"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Multi-Step Chain (Translate then Summarize)")
    print("="*60)
    
    model = ChatOpenAI(model="gpt-4o-mini", api_key="your-api-key-here")
    
    # Step 1: Translation chain
    translate_prompt = ChatPromptTemplate.from_template(
        "Translate the following text to Spanish: {text}"
    )
    translate_chain = translate_prompt | model | StrOutputParser()
    
    # Step 2: Summarization chain
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize this text in 1 sentence: {text}"
    )
    summarize_chain = summarize_prompt | model | StrOutputParser()
    
    # Step 3: Combine chains
    full_chain = translate_chain | summarize_chain
    
    # Step 4: Use it
    english_text = "Python is a great programming language for beginners."
    result = full_chain.invoke({"text": english_text})
    
    print(f"Original: {english_text}")
    print(f"Result (translated then summarized): {result}")
    
    return result


# ============================================================================
# 🎨 PART 8: STRUCTURED OUTPUT
# ============================================================================

def example_6_structured_output():
    """Example 6: Get structured data from model"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Structured Output (JSON)")
    print("="*60)
    
    from pydantic import BaseModel, Field
    
    # Define the structure you want
    class PersonInfo(BaseModel):
        name: str = Field(description="Person's name")
        age: int = Field(description="Person's age")
        city: str = Field(description="Person's city")
    
    # Create model with structured output
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="your-api-key-here",
    ).with_structured_output(PersonInfo)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(
        "Extract information about this person: {description}"
    )
    
    # Create chain
    chain = prompt | model
    
    # Use it
    description = "John is 30 years old and lives in New York."
    result = chain.invoke({"description": description})
    
    print(f"Input: {description}")
    print(f"Structured Output: {result}")
    print(f"Type: {type(result)}")  # Should be PersonInfo object
    
    return result


# ============================================================================
# 📚 PART 9: COMMON PATTERNS CHEAT SHEET
# ============================================================================

"""
CHEAT SHEET: Common LangChain Patterns
======================================

1. SIMPLE CHAT:
   prompt | model | StrOutputParser()

2. WITH VARIABLES:
   prompt = ChatPromptTemplate.from_template("Hello {name}")
   chain.invoke({"name": "Alice"})

3. MULTI-MESSAGE PROMPT:
   ChatPromptTemplate.from_messages([
       ("system", "You are helpful"),
       ("user", "{question}")
   ])

4. STREAMING (for real-time responses):
   for chunk in chain.stream({"input": "hello"}):
       print(chunk)

5. BATCH PROCESSING:
   chain.batch([{"input": "text1"}, {"input": "text2"}])

6. ASYNC (for concurrent requests):
   await chain.ainvoke({"input": "hello"})

7. WITH TOOLS (for function calling):
   model.bind_tools([tool1, tool2])

8. WITH RETRIEVAL (RAG):
   retriever | prompt | model | parser
"""


# ============================================================================
# 🤖 PART 10: AGENTS - LangChain vs LangGraph
# ============================================================================

"""
LANGCHAIN vs LANGGRAPH FOR AGENTS:
==================================

IMPORTANT: You can build agents with BOTH, but they serve different purposes!

┌─────────────────────────────────────────────────────────────┐
│ LANGCHAIN (Higher-level, easier)                            │
├─────────────────────────────────────────────────────────────┤
│ ✅ Use create_agent() for most agent needs                  │
│ ✅ Built on top of LangGraph (you get LangGraph benefits)   │
│ ✅ Simple API: create_agent(model, tools)                   │
│ ✅ Good for: Standard agent patterns, quick prototyping     │
│ ✅ No need to learn LangGraph for basic agents             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ LANGGRAPH (Lower-level, more control)                       │
├─────────────────────────────────────────────────────────────┤
│ ✅ Use when you need deep customization                     │
│ ✅ Full control over agent workflow (nodes, edges, state)   │
│ ✅ Good for: Complex workflows, custom logic, fine control  │
│ ✅ When: Standard agent patterns aren't enough              │
└─────────────────────────────────────────────────────────────┘

KEY INSIGHT:
============
LangChain's create_agent() IS built on LangGraph!
So you get LangGraph benefits (durable execution, streaming, etc.)
without needing to learn LangGraph directly.

RECOMMENDATION:
===============
1. Start with LangChain's create_agent() ← EASIER
2. If you need more control → Use LangGraph directly
"""


def example_agent_langchain():
    """Example: Build agent using LangChain (easier way)"""
    print("\n" + "="*60)
    print("EXAMPLE: Agent with LangChain create_agent()")
    print("="*60)
    
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    
    # Step 1: Define tools (functions the agent can use)
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny, 72°F"
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression"""
        return str(eval(expression))  # In production, use safer eval
    
    # Step 2: Create model
    model = ChatOpenAI(model="gpt-4o-mini", api_key="your-api-key")
    
    # Step 3: Create agent (this uses LangGraph under the hood!)
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate],
    )
    
    # Step 4: Use the agent
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What's the weather in Tokyo and what's 15 * 23?"}]
    })
    
    print(f"Agent Response: {response['messages'][-1].content}")
    
    return response


def example_agent_langgraph():
    """Example: Build agent using LangGraph directly (more control)"""
    print("\n" + "="*60)
    print("EXAMPLE: Agent with LangGraph (advanced customization)")
    print("="*60)
    
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    # Step 1: Create model
    model = ChatOpenAI(model="gpt-4o-mini", api_key="your-api-key")
    
    # Step 2: Define nodes (steps in your agent workflow)
    def call_model(state: MessagesState):
        """Node: Call the LLM"""
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    
    def check_tools(state: MessagesState):
        """Node: Check if tools need to be called"""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"  # Go to tools node
        return "end"  # Finish
    
    # Step 3: Build the graph (define workflow)
    workflow = StateGraph(MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    workflow.add_conditional_edges("model", check_tools)
    workflow.add_edge("tools", END)  # Simplified - in real app, add tool execution
    workflow.add_edge("end", END)
    
    # Step 4: Compile the graph
    agent = workflow.compile()
    
    # Step 5: Use it
    response = agent.invoke({
        "messages": [HumanMessage(content="Hello!")]
    })
    
    print(f"Agent Response: {response['messages'][-1].content}")
    
    return response


"""
WHEN TO USE WHICH:
==================

Use LangChain create_agent() when:
- ✅ You want to build agents quickly
- ✅ Standard agent patterns work for you
- ✅ You don't need custom workflow logic
- ✅ You want simplicity

Use LangGraph directly when:
- ✅ You need custom workflow logic
- ✅ You need fine-grained control over state
- ✅ You need custom nodes/edges
- ✅ Standard agent patterns aren't enough
- ✅ You need deterministic + agentic workflows mixed
"""


# ============================================================================
# 🚀 PART 11: WHAT CAN YOU BUILD?
# ============================================================================

"""
WHAT YOU CAN BUILD WITH LANGCHAIN + OPENAI:
===========================================

1. ✅ CHATBOTS
   - Customer support bots
   - Personal assistants
   - Interactive Q&A systems

2. ✅ SUMMARIZATION
   - Article summaries
   - Meeting notes
   - Long document summaries

3. ✅ QUESTION-ANSWERING
   - Document Q&A
   - Knowledge bases
   - RAG (Retrieval Augmented Generation)

4. ✅ TEXT PROCESSING
   - Translation
   - Sentiment analysis
   - Text classification
   - Content generation

5. ✅ AGENTS
   - Tools-using agents
   - Multi-step reasoning
   - Autonomous systems

6. ✅ DATA EXTRACTION
   - Structured data from text
   - Entity extraction
   - Form filling

7. ✅ CODE GENERATION
   - Code assistants
   - Documentation generators
   - Code review tools
"""


# ============================================================================
# 🎓 PART 11: KEY LEARNINGS
# ============================================================================

"""
KEY TAKEAWAYS:
=============

1. LCEL (LangChain Expression Language) uses | to chain components
   Example: prompt | model | parser

2. ChatPromptTemplate is for multi-message prompts (system, user, assistant)
   Use this for chat models like GPT-4

3. StrOutputParser converts AI responses to strings
   Always use this unless you need structured output

4. Temperature controls creativity:
   - 0.0 = Deterministic, factual
   - 0.7 = Balanced (default)
   - 1.0 = Creative, varied

5. Always set your API key securely:
   - Use environment variables: os.environ["OPENAI_API_KEY"]
   - Or pass directly: ChatOpenAI(api_key="...")

6. Chains are reusable:
   Create once, invoke many times with different inputs

7. Streaming for better UX:
   Use .stream() instead of .invoke() for real-time responses
"""


# ============================================================================
# 🧪 TEST YOUR KNOWLEDGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 LANGCHAIN + OPENAI TUTORIAL")
    print("="*60)
    print("\nUncomment the examples below to try them!")
    print("(Make sure to set your OPENAI_API_KEY first)\n")
    
    # Uncomment to run examples:
    # example_1_simple_chat()
    # example_2_summarization()
    # example_3_chatbot()
    # example_4_document_qa()
    # example_5_multi_step()
    # example_6_structured_output()
    
    print("\n✅ Tutorial complete! You now know LangChain basics!")

