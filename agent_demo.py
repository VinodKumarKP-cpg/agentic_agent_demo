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

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

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
                logger.error("MCP client not available. Please install the required dependencies.")
                raise AgentDemoError("MCP client not available. Please install mcp_client package.")
            except Exception as e:
                logger.error(f"Failed to initialize MCP tools: {e}")
                raise AgentDemoError(f"Failed to initialize tools: {e}")
        return self._tools


class AgentService:
    """Service class for managing the AI agent and processing requests"""

    def __init__(self):
        self.llm_manager = BedrockLLMManager()
        self.tool_manager = ToolManager([{
            'name': 'git_utils',
            'url': 'http://localhost:8000/mcp'
        },
            {
                'name': 'env_lookup',
                'url': 'http://localhost:8005/mcp'
            },
            {
                'name': 'weather_lookup',
                'url': 'http://localhost:8020/mcp'
            }
        ])
        self._agent = None

    async def initialize_agent(self) -> None:
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

    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return agent response"""
        if not self._agent:
            await self.initialize_agent()

        try:
            response = await self._agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            logger.info("Agent request processed successfully")
            return response
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            raise AgentDemoError(f"Failed to process request: {e}")


def format_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Format tool call for display"""
    return {
        "Tool": tool_call.get('name', 'Unknown'),
        "Arguments": tool_call.get('args', {}),
        "ID": tool_call.get('id', 'N/A')
    }


def display_trace_information(messages: List[Any]) -> None:
    """Display detailed trace information in an organized manner"""
    st.subheader("🔍 Execution Trace")

    with st.expander("View Detailed Tool Calls and Reasoning", expanded=False):
        for i, message in enumerate(messages):
            st.markdown(f"### Step {i + 1}: {message.type.title()}")

            # Display tool calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                st.markdown("**🔧 Tool Calls:**")
                for j, tool_call in enumerate(message.tool_calls):
                    with st.container():
                        st.markdown(f"*Tool Call {j + 1}:*")
                        formatted_call = format_tool_call(tool_call)
                        st.json(formatted_call)

            # Display message content
            if hasattr(message, 'content') and message.content:
                st.markdown("**💭 Response Content:**")
                st.markdown(message.content)

            if i < len(messages) - 1:
                st.divider()


def display_conversation(messages: List[Any]) -> None:
    """Display the conversation in a chat-like format"""
    st.subheader("💬 Conversation")

    for message in messages:
        if message.type.lower() == 'human':
            with st.chat_message("user"):
                st.write(message.content)
        elif message.type.lower() == 'ai' and hasattr(message, 'content') and message.content:
            with st.chat_message("assistant"):
                st.markdown(message.content)


def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Professional AI Agent Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Header
    st.title("🤖 Professional AI Agent Demo")
    st.markdown("---")
    st.markdown("**Powered by Amazon Bedrock & Claude 3.5 Sonnet**")

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
                placeholder="Ask me anything! I have access to various tools to help you.",
                height=100
            )
            col1, col2, col3 = st.columns([1, 1, 4])

            with col1:
                submit_button = st.form_submit_button("Send", type="primary")
            with col2:
                clear_button = st.form_submit_button("Clear History")

        # Handle form submissions
        if clear_button:
            st.session_state.conversation_history = []
            st.rerun()

        if submit_button and user_input.strip():
            with st.spinner("🤔 Processing your request..."):
                try:
                    # Update agent configuration if changed
                    if (selected_model != st.session_state.agent_service.llm_manager.model_id or
                            selected_region != st.session_state.agent_service.llm_manager.region):
                        st.session_state.agent_service = AgentService()
                        st.session_state.agent_service.llm_manager.model_id = selected_model
                        st.session_state.agent_service.llm_manager.region = selected_region

                    # Process the request
                    response = asyncio.run(
                        st.session_state.agent_service.process_request(user_input)
                    )

                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        'input': user_input,
                        'response': response,
                        'timestamp': st.cache_data.clear()  # This will help with memory management
                    })

                    st.success("✅ Response generated successfully!")

                except AgentDemoError as e:
                    st.error(f"❌ Agent Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {e}")
                    logger.error(f"Unexpected error: {e}")

    # Display results
    if st.session_state.conversation_history:
        st.markdown("---")

        # Show latest interaction
        latest = st.session_state.conversation_history[-1]

        # Display trace information
        display_trace_information(latest['response']['messages'])

        st.markdown("---")

        # Display conversation
        display_conversation(latest['response']['messages'])

        # Show conversation history
        if len(st.session_state.conversation_history) > 1:
            st.markdown("---")
            st.subheader("📚 Conversation History")

            with st.expander("View Previous Conversations", expanded=False):
                for i, conv in enumerate(reversed(st.session_state.conversation_history[:-1])):
                    st.markdown(f"**Conversation {len(st.session_state.conversation_history) - i - 1}:**")
                    st.markdown(f"*User:* {conv['input'][:100]}...")
                    if conv['response']['messages']:
                        final_message = conv['response']['messages'][-1]
                        if hasattr(final_message, 'content') and final_message.content:
                            st.markdown(f"*Assistant:* {final_message.content[:100]}...")
                    st.divider()

    # Footer
    st.markdown("---")
    st.markdown(
        "*Built with Streamlit, Amazon Bedrock, and Claude 3.5 Sonnet. "
        "This demo showcases professional AI agent capabilities with tool integration.*"
    )


if __name__ == "__main__":
    main()