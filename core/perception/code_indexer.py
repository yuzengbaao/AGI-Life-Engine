import os
import ast
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

# Configure logging
logger = logging.getLogger("CodeIndexer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class SymbolNode:
    """Represents a code symbol (Class, Function, Method)."""
    def __init__(self, name: str, type: str, file_path: str, line: int, parent: Optional[str] = None, bases: Optional[List[str]] = None):
        self.name = name
        self.type = type  # 'class', 'function', 'method'
        self.file_path = file_path
        self.line = line
        self.parent = parent  # e.g., class name for a method
        self.bases = bases or []  # Base classes for inheritance

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "file": os.path.basename(self.file_path),
            "line": self.line,
            "parent": self.parent,
            "bases": self.bases
        }

    def __repr__(self) -> str:
        return f"<{self.type} {self.name} @ {os.path.basename(self.file_path)}:{self.line}>"

class ProjectIndexer:
    """
    The 'Cortex' of the AGI.
    Scans the codebase to build a semantic map of symbols and their relationships.
    """
    EXCLUDED_DIRS: Set[str] = {
        'venv', '__pycache__', 'backups', 'logs', 'data',
        'dist', 'build', '.git', '.pytest_cache', '__pycache__'
    }
    PYTHON_EXT = '.py'

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        # symbol_name -> list of definitions (handling duplicates/overloads)
        self.definitions: Dict[str, List[SymbolNode]] = defaultdict(list)
        # symbol_name -> list of usage sites
        self.calls: Dict[str, List[Dict]] = defaultdict(list)
        self.indexed_files = 0

    def build_index(self) -> None:
        """Scans the project directory and builds the AST index."""
        logger.info(f"Starting index build for: {self.root_dir}")
        self.indexed_files = 0
        self.definitions.clear()
        self.calls.clear()

        try:
            for root, dirs, files in os.walk(self.root_dir):
                # Filter out unwanted directories in-place
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in self.EXCLUDED_DIRS]

                python_files = [f for f in files if f.endswith(self.PYTHON_EXT)]
                for file in python_files:
                    file_path = Path(root) / file
                    self._index_file(file_path)

            logger.info(f"Index build complete. Scanned {self.indexed_files} files.")
            logger.info(f"Knowledge Graph: {len(self.definitions)} unique symbols defined.")

        except Exception as e:
            logger.error(f"Error during indexing process: {e}", exc_info=True)

    def _index_file(self, file_path: Path) -> None:
        """Index a single Python file using AST parsing."""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))

            visitor = IndexVisitor(str(file_path), self)
            visitor.visit(tree)
            self.indexed_files += 1

        except SyntaxError as se:
            logger.warning(f"Syntax error in {file_path.name} at line {se.lineno}: {se.msg}")
        except PermissionError:
            logger.warning(f"Permission denied reading {file_path.name}. Skipping.")
        except UnicodeDecodeError:
            logger.warning(f"Encoding error reading {file_path.name}. Skipping non-UTF-8 file.")
        except Exception as e:
            logger.warning(f"Unexpected error parsing {file_path.name}: {e}")

    def get_symbol_location(self, name: str) -> List[SymbolNode]:
        """Finds where a symbol is defined."""
        return self.definitions.get(name, [])

    def get_callers(self, symbol_name: str) -> List[Dict]:
        """Finds who calls this symbol (Dependency tracing)."""
        return self.calls.get(symbol_name, [])


class IndexVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, indexer: ProjectIndexer):
        self.file_path = file_path
        self.indexer = indexer
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle cases like 'super.Module.Base'
                names = []
                n = base
                while isinstance(n, ast.Attribute):
                    names.append(n.attr)
                    n = n.value
                if isinstance(n, ast.Name):
                    names.append(n.id)
                bases.append('.'.join(reversed(names)))

        symbol = SymbolNode(node.name, 'class', self.file_path, node.lineno, bases=bases)
        self.indexer.definitions[node.name].append(symbol)

        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_method = bool(self.current_class)
        type_ = 'method' if is_method else 'function'
        name = node.name
        parent = self.current_class

        symbol = SymbolNode(name, type_, self.file_path, node.lineno, parent=parent)

        # Index by simple name (e.g., 'fail_goal')
        self.indexer.definitions[name].append(symbol)

        # Index by Qualified Name (e.g., 'GoalManager.fail_goal') only for methods
        if is_method:
            qname = f"{parent}.{name}"
            self.indexer.definitions[qname].append(symbol)

        prev_func = self.current_function
        self.current_function = name
        self.generic_visit(node)
        self.current_function = prev_func

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._get_func_name(node.func)
        if func_name:
            context_parts = []
            if self.current_class:
                context_parts.append(self.current_class)
            if self.current_function:
                context_parts.append(self.current_function)
            context = ".".join(context_parts) if context_parts else "module_level"

            call_info = {
                'file': os.path.basename(self.file_path),
                'line': node.lineno,
                'context': context
            }
            self.indexer.calls[func_name].append(call_info)

        self.generic_visit(node)

    def _get_func_name(self, node) -> Optional[str]:
        """Extract the called function's name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Return only the final attribute name (method name)
            return node.attr
        return None