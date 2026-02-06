"""Source code loader for DM/apriori.py"""

from pathlib import Path


class SourceCodeModule:
    """A class that displays source code when printed."""
    
    def __init__(self, name: str, source_path: Path):
        self.name = name
        self._source_path = source_path
        self._source_code = None
    
    @property
    def source_code(self) -> str:
        """Lazily load source code."""
        if self._source_code is None:
            self._source_code = self._source_path.read_text(encoding='utf-8')
        return self._source_code
    
    def __repr__(self) -> str:
        return self.source_code
    
    def __str__(self) -> str:
        return self.source_code


# Get the path to the source file
_source_file = Path(__file__).parent / "sources" / "apriori.py"
apriori = SourceCodeModule("DM.apriori", _source_file)
