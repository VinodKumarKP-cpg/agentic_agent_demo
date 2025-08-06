import logging
import os
import re
import shutil
import subprocess
import sys
from typing import List, Dict, Any, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPServerConfig:
    """Configuration for a single MCP server"""

    def __init__(self, name: str,
                 command: Optional[str] = None,
                 args: Optional[List[str]] = None,
                 description: Optional[str] = None,
                 title: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None,
                 url: Optional[str] = None,
                 transport: Optional[str] = "stdio",
                 **kwargs):
        self.name = name
        self.command = command
        self.args = args
        self.description = description or f"MCP Server: {name}"
        self.title = title
        self.env = env or {}
        self.url = url
        self.transport = transport
        self.additional_config = kwargs if kwargs else {}

        root_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # Validate command exists
        if command and not os.path.exists(command):
            raise ValueError(f"Command {command} does not exist for server {name}")

        # Validate script files exist
        if args and 'run' not in args:
            for index, script in enumerate(args):
                if root_directory not in script:
                    script = os.path.join(root_directory, script)
                    args[index] = script
                if not os.path.exists(script):
                    raise ValueError(f"Server script {script} does not exist for server {name}")

    def get_config(self) -> dict[str, dict[str, str | list[str] | None | dict[str, str] | dict[Any, Any]]] | dict[
        str, dict[str, str | None]] | None:
        """Return the server configuration as a dictionary"""

        config = {
            self.name: self.additional_config['kwargs']
        }

        if self.command is not None:
            config[self.name].update({
                "command": self.command,
                "args": self.args,
                "description": self.description,
                "transport": self.transport,
                "title": self.title,
                "env": self.env
            })
        elif self.url is not None:
            config[self.name].update({
                "url": self.url,
                "description": self.description,
                "transport": self.transport,
                "title": self.title,
                "env": self.env
            })
        return config


class MCPClient:
    def __init__(self, server_config):
        self.mcp_initialized = False
        self.server_configs: Dict = {}
        self.add_servers(server_config)
        self.client = None

    def add_server(self,
                   name: str,
                   command: Optional[str] = None,
                   args: Optional[List[str]] = None,
                   description: Optional[str] = None,
                   title: Optional[str] = None,
                   env: Optional[Dict[str, str]] = None,
                   url: Optional[str] = None,
                   transport: Optional[str] = "stdio",
                   **kwargs):
        """Add an MCP server configuration"""
        try:
            config = MCPServerConfig(name, command, args, description, title, env, url, transport, **kwargs)
            self.server_configs.update(config.get_config())
            logger.info(f"Added MCP server configuration: {name}")
        except ValueError as e:
            logger.error(f"Failed to add server {name}: {e}")
            raise

    def resolve_env_vars(self, config_dict):
        """Replace values starting with '$' with environment variable values."""
        resolved = {}
        for k, v in config_dict.items():
            if isinstance(v, str) and v.startswith('$'):
                match = re.match(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)', v)
                var_name = match.group(1) or match.group(2) if match else None
                resolved[k] = os.environ.get(var_name, '') if var_name else v
            else:
                resolved[k] = v
        return resolved

    def add_servers(self, servers: List[Dict[str, Any]]):
        """Add multiple MCP server configurations

        Args:
            servers: List of server configs, each with keys: name, command, args, description (optional)
        """
        for server_config in servers:
            server_config['env'] = self.resolve_env_vars(server_config.get('env', {}))
            server_config['headers'] = self.resolve_env_vars(server_config.get('headers', {}))

            env = os.environ.copy()
            env.update(server_config.get('env', {}))
            env.update(server_config.get('headers', {}))
            transport = 'stdio'
            if 'url' in server_config:
                url = server_config['url']
                if 'mcp' in url:
                    transport = 'streamable_http'
                    server_config.setdefault('headers', {})
                    server_config['headers'].update(server_config.get('env', {}))
                elif 'sse' in url:
                    transport = 'sse'
                else:
                    raise ValueError("Unsupported url. It should either end with mcp and sse")

            self.add_server(
                name=server_config['name'],
                command=self.which(server_config['command']) if transport == 'stdio' else None,
                args=server_config['args'] if transport == 'stdio' else None,
                description=server_config.get('description'),
                title=server_config.get('title', server_config['name']),
                env=env,
                url=server_config['url'] if transport in ['streamable_http', "sse"] else None,
                transport=transport,
                kwargs=server_config
            )

    async def setup_mcp_client(self):
        """Initialize a MultiServerMCPClient with the provided server configuration."""
        self.client = MultiServerMCPClient(self.server_configs)

    async def get_tools(self) -> List[Any]:
        tools = await self.client.get_tools()
        return tools

    def which(self, program):
        """Cross-platform implementation of 'which' command"""
        # First try shutil.which (available in Python 3.3+)
        path = shutil.which(program)
        if path:
            return path

        # Fallback for older Python versions or edge cases
        if sys.platform.startswith('win'):
            # Windows-specific implementation
            return self._which_windows(program)
        else:
            # Unix-like systems
            return self._which_unix(program)

    def _which_windows(self, program):
        """Windows-specific implementation of which"""
        # Add common executable extensions if not present
        if not any(program.lower().endswith(ext) for ext in ['.exe', '.bat', '.cmd', '.com']):
            # Try with different extensions
            for ext in ['.exe', '.bat', '.cmd', '.com']:
                extended_program = program + ext
                path = shutil.which(extended_program)
                if path:
                    return path

        # If shutil.which didn't work, try manual search
        paths = os.environ.get('PATH', '').split(os.pathsep)

        for path_dir in paths:
            if not path_dir:
                continue

            # Try the program as-is
            full_path = os.path.join(path_dir, program)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                return full_path

            # Try with executable extensions
            for ext in ['.exe', '.bat', '.cmd', '.com']:
                full_path_with_ext = full_path + ext
                if os.path.isfile(full_path_with_ext) and os.access(full_path_with_ext, os.X_OK):
                    return full_path_with_ext

        raise RuntimeError(f"'{program}' is not found in the system path.")

    def _which_unix(self, program):
        """Unix-like implementation of which"""
        try:
            result = subprocess.run(['which', program], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            raise RuntimeError(f"'{program}' is not found in the system path.")
