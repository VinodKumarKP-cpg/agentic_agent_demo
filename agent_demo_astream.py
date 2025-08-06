import asyncio
import logging
from typing import Dict, List, Any, Optional
import streamlit as st
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrock
from langgraph.prebuilt import create_react_agent
import nest_asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()


# Global event loop for async operations
def get_or_create_eventloop():
    """Get or create an event loop for async operations"""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentDemoError(Exception):
    """Custom exception for agent demo errors"""
    pass


class BedrockLLMManager:
    """Manages Bedrock LLM initialization and configuration"""

    def __init__(self, region: str = 'us-east-1', model_id: str = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'):
        self.region = region
        self.model_id = model_id
        self._client = None
        self._llm = None

    def get_bedrock_client(self) -> boto3.client:
        """Initialize and return Bedrock client with proper configuration"""
        if self._client is None:
            try:
                self._client = boto3.client(
                    'bedrock-runtime',
                    region_name=self.region,
                    config=Config(
                        read_timeout=1000,
                        retries={'max_attempts': 3}
                    )
                )
                logger.info(f"Bedrock client initialized for region: {self.region}")
            except ClientError as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise AgentDemoError(f"Failed to initialize Bedrock client: {e}")
        return self._client

    def get_llm(self) -> ChatBedrock:
        """Initialize and return ChatBedrock LLM"""
        if self._llm is None:
            try:
                self._llm = ChatBedrock(
                    client=self.get_bedrock_client(),
                    model_kwargs={
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "top_p": 0.9
                    },
                    model=self.model_id
                )
                logger.info(f"ChatBedrock LLM initialized with model: {self.model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatBedrock LLM: {e}")
                raise AgentDemoError(f"Failed to initialize LLM: {e}")
        return self._llm


class ToolManager:
    """Manages MCP tool initialization and retrieval"""

    def __init__(self, server_configs: List[Dict[str, str]]):
        self.server_configs = server_configs
        self._tools = None
        self._client = None

    async def initialize_tools(self) -> List[Any]:
        """Initialize and return MCP tools"""
        if self._tools is None:
            try:
                # Import here to avoid import issues if MCP client is not available
                from mcp_client import MCPClient

                self._client = MCPClient(server_config=self.server_configs)
                await self._client.setup_mcp_client()
                self._tools = await self._client.get_tools()
                logger.info(f"MCP tools initialized: {len(self._tools)} tools available")
            except ImportError:
                logger.warning("MCP client not available. Running without external tools.")
                # Provide mock tools for demo purposes
                self._tools = []
            except Exception as e:
                logger.error(f"Failed to initialize MCP tools: {e}")
                # Provide mock tools for demo purposes
                self._tools = []
        return self._tools


class StreamingDisplay:
    """Handles real-time display updates for streaming responses"""

    def __init__(self):
        self.step_counter = 0
        self.current_step_container = None
        self.current_content_container = None
        self.tool_calls_container = None
        self.thinking_container = None

    def create_new_step(self, step_type: str, step_name: str = None) -> None:
        """Create a new step in the streaming display"""
        self.step_counter += 1

        if step_name:
            step_title = f"Step {self.step_counter}: {step_name}"
        else:
            step_title = f"Step {self.step_counter}: {step_type.title()}"

        self.current_step_container = st.expander(
            f"🔄 {step_title}",
            expanded=True
        )

        with self.current_step_container:
            if step_type == "ai":
                self.thinking_container = st.empty()
                self.tool_calls_container = st.empty()
                self.current_content_container = st.empty()
            elif step_type == "tool":
                self.current_content_container = st.empty()

    def update_thinking(self, content: str) -> None:
        """Update the thinking/reasoning display"""
        if self.thinking_container:
            with self.thinking_container.container():
                st.markdown("**🧠 Thinking:**")
                st.markdown(f"_{content}_")

    def update_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Update tool calls display"""
        if self.tool_calls_container and tool_calls:
            with self.tool_calls_container.container():
                st.markdown("**🔧 Using Tools:**")
                for i, tool_call in enumerate(tool_calls):
                    with st.container():
                        st.markdown(f"*Tool {i + 1}: {tool_call.get('name', 'Unknown')}*")
                        if tool_call.get('args'):
                            st.json(tool_call['args'])

    def update_content(self, content: str, is_final: bool = False) -> None:
        """Update content display"""
        if self.current_content_container:
            with self.current_content_container.container():
                if is_final:
                    st.markdown("**✅ Final Response:**")
                    st.markdown(content)
                else:
                    st.markdown("**💭 Response (streaming):**")
                    st.markdown(content + "▋")  # Add cursor for streaming effect

    def update_tool_result(self, tool_name: str, result: str) -> None:
        """Update tool result display"""
        if self.current_content_container:
            with self.current_content_container.container():
                st.markdown(f"**🔧 {tool_name} Result:**")
                with st.code():
                    st.text(result[:500] + "..." if len(result) > 500 else result)


class AgentService:
    """Service class for managing the AI agent and processing requests"""

    def __init__(self):
        self.llm_manager = BedrockLLMManager()
        self.tool_manager = ToolManager([{
            'name': 'git_utils',
            'url': 'http://localhost:8000/mcp'
        }])
        self._agent = None

    def initialize_agent_sync(self) -> None:
        """Synchronous wrapper for agent initialization"""
        loop = get_or_create_eventloop()
        loop.run_until_complete(self._initialize_agent_async())

    async def _initialize_agent_async(self) -> None:
        """Initialize the React agent with LLM and tools"""
        try:
            llm = self.llm_manager.get_llm()
            tools = await self.tool_manager.initialize_tools()

            system_prompt = """You are a professional AI assistant with access to various tools.

            Guidelines:
            - Always be helpful, accurate, and professional
            - Use tools when necessary to provide better assistance
            - Explain your reasoning and tool usage clearly
            - If you encounter errors, explain them clearly to the user
            - Provide structured and well-formatted responses
            """

            self._agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=system_prompt
            )
            logger.info("React agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise AgentDemoError(f"Failed to initialize agent: {e}")

    def stream_request_sync(self, user_input: str, display: StreamingDisplay) -> Dict[str, Any]:
        """Process user input with real-time streaming updates"""
        if not self._agent:
            self.initialize_agent_sync()

        try:
            all_messages = []
            processed_message_ids = set()

            # Create event loop for this operation
            loop = get_or_create_eventloop()

            # Create the async generator
            async def stream_generator():
                async for chunk in self._agent.astream({
                    "messages": [{"role": "user", "content": user_input}]
                }):
                    yield chunk

            # Process the stream synchronously
            gen = stream_generator()

            try:
                while True:
                    try:
                        # Get next chunk from async generator
                        chunk = loop.run_until_complete(gen.__anext__())

                        # Process each chunk from the stream
                        if 'agent' in chunk and 'messages' in chunk['agent']:
                            messages = chunk['agent']['messages']

                            for message in messages:
                                # Create a unique ID for message deduplication
                                message_id = f"{message.type}_{getattr(message, 'id', id(message))}"

                                # Skip if we've already processed this message
                                if message_id in processed_message_ids:
                                    continue

                                processed_message_ids.add(message_id)
                                all_messages.append(message)

                                # Handle different message types
                                if message.type == 'ai':
                                    # Create new step for AI response
                                    display.create_new_step("ai", "AI Processing")

                                    # Show tool calls if present
                                    if hasattr(message, 'tool_calls') and message.tool_calls:
                                        display.update_tool_calls(message.tool_calls)

                                    # Show content if present
                                    if hasattr(message, 'content') and message.content:
                                        display.update_content(message.content, is_final=True)

                                elif message.type == 'tool':
                                    # Create new step for tool execution
                                    tool_name = getattr(message, 'name', 'Unknown Tool')
                                    display.create_new_step("tool", f"Running {tool_name}")

                                    # Show tool result
                                    if hasattr(message, 'content'):
                                        display.update_tool_result(tool_name, str(message.content))

                                elif message.type == 'human':
                                    # User message - add to messages but don't display
                                    continue

                                # Small delay for visual effect
                                time.sleep(0.1)

                    except StopAsyncIteration:
                        # End of stream
                        break

            finally:
                # Properly close the async generator
                try:
                    loop.run_until_complete(gen.aclose())
                except Exception:
                    pass

            logger.info(f"Streaming request processed successfully. Total messages: {len(all_messages)}")
            return {"messages": all_messages}

        except Exception as e:
            logger.error(f"Failed to stream request: {e}")
            raise AgentDemoError(f"Failed to stream request: {e}")


def display_final_conversation(messages: List[Any]) -> None:
    """Display the final conversation in a clean chat format"""
    st.subheader("💬 Final Conversation")

    user_messages = []
    ai_messages = []

    # Separate and collect messages
    for message in messages:
        if message.type.lower() == 'human':
            user_messages.append(message)
        elif message.type.lower() == 'ai' and hasattr(message, 'content') and message.content:
            ai_messages.append(message)

    # Display user message first
    for message in user_messages:
        with st.chat_message("user"):
            st.write(message.content)

    # Display final AI response (usually the last AI message with content)
    if ai_messages:
        final_ai_message = ai_messages[-1]  # Get the last AI message
        with st.chat_message("assistant"):
            st.markdown(final_ai_message.content)


def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Real-time AI Agent Demo",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for better streaming appearance
    st.markdown("""
    <style>
    .streaming-container {
        border-left: 3px solid #ff6b6b;
        padding-left: 10px;
        margin: 5px 0;
    }

    .step-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🚀 Real-time AI Agent Demo")
    st.markdown("---")
    st.markdown("**Watch the AI agent think and work in real-time!**")

    # Initialize session state
    if 'agent_service' not in st.session_state:
        st.session_state.agent_service = AgentService()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        model_options = [
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        ]
        selected_model = st.selectbox("Select Model:", model_options)

        # Region selection
        region_options = ["us-east-1", "us-west-2", "eu-west-1"]
        selected_region = st.selectbox("Select Region:", region_options)

        st.markdown("---")
        st.markdown("**💡 Streaming Features:**")
        st.markdown("- Real-time step tracking")
        st.markdown("- Live tool execution")
        st.markdown("- Progressive response display")

        if st.button("Reset Conversation"):
            st.session_state.conversation_history = []
            st.rerun()

    # Main interaction area
    with st.container():
        st.subheader("🗨️ Chat Interface")

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Enter your message:",
                placeholder="Ask me anything! Watch as I think through the problem step by step.",
                height=100
            )

            col1, col2, col3 = st.columns([1, 1, 4])

            with col1:
                submit_button = st.form_submit_button("🚀 Send", type="primary")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear")

        # Handle form submissions
        if clear_button:
            st.session_state.conversation_history = []
            st.rerun()

        if submit_button and user_input.strip():
            # Show user message immediately
            with st.chat_message("user"):
                st.write(user_input)

            # Create streaming container
            streaming_container = st.container()

            with streaming_container:
                st.markdown("### 🔄 Live Agent Processing")

                # Initialize streaming display
                display = StreamingDisplay()

                try:
                    # Update agent configuration if changed
                    if (selected_model != st.session_state.agent_service.llm_manager.model_id or
                            selected_region != st.session_state.agent_service.llm_manager.region):
                        st.session_state.agent_service = AgentService()
                        st.session_state.agent_service.llm_manager.model_id = selected_model
                        st.session_state.agent_service.llm_manager.region = selected_region

                    # Process the request with streaming
                    with st.spinner("🤖 Initializing agent..."):
                        response = st.session_state.agent_service.stream_request_sync(user_input, display)

                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        'input': user_input,
                        'response': response,
                        'timestamp': time.time()
                    })

                    st.success("✅ Processing complete!")

                    # Debug: Show message count
                    st.info(f"📊 Processed {len(response['messages'])} messages")

                except AgentDemoError as e:
                    st.error(f"❌ Agent Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {e}")
                    logger.error(f"Unexpected error: {e}")

    # Display final results
    if st.session_state.conversation_history:
        st.markdown("---")

        # Show latest final conversation
        latest = st.session_state.conversation_history[-1]

        # Debug info
        if st.checkbox("🔍 Show Debug Info", key="debug_info"):
            st.write("**Debug - Raw Response:**")
            st.json({
                "message_count": len(latest['response']['messages']),
                "message_types": [msg.type for msg in latest['response']['messages']],
                "has_content": [hasattr(msg, 'content') and bool(msg.content) for msg in latest['response']['messages']]
            })

        # Only display if we have messages
        if latest['response']['messages']:
            display_final_conversation(latest['response']['messages'])
        else:
            st.warning("⚠️ No messages to display in final conversation")

        # Show conversation history
        if len(st.session_state.conversation_history) > 1:
            st.markdown("---")
            st.subheader("📚 Previous Conversations")

            with st.expander("View Conversation History", expanded=False):
                for i, conv in enumerate(reversed(st.session_state.conversation_history[:-1])):
                    st.markdown(f"**Conversation {len(st.session_state.conversation_history) - i - 1}:**")

                    # Show user input
                    with st.chat_message("user"):
                        st.write(conv['input'])

                    # Show final AI response
                    if conv['response']['messages']:
                        for message in conv['response']['messages']:
                            if message.type.lower() == 'ai' and hasattr(message, 'content') and message.content:
                                with st.chat_message("assistant"):
                                    st.markdown(message.content)
                                break

                    st.divider()

    # Footer
    st.markdown("---")
    st.markdown(
        "*Real-time AI Agent Demo - Built with Streamlit, Amazon Bedrock, and Claude 3.5 Sonnet. "
        "Experience transparent AI decision-making with live streaming updates.*"
    )


if __name__ == "__main__":
    main()