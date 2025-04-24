from typing import Any, Dict, List

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream


@component
class PreWriterEnricher:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict[str, List[Document]]:
        """Declare output type as a list of documents"""
        for doc in documents:
            doc.meta["input_file_audio_path"] = self.audio_path
            doc.meta["flags"] = {
                "text_embedded": False,
                "voice_embedded": False,
                "has_summary": False,
                "has_fingerprint": False,
            }
            with open(self.audio_path, "rb") as f:
                raw = f.read()
            doc.blob = ByteStream(data=raw, meta={"source": self.audio_path}, mime_type="audio/mpeg")
        return {"documents": documents}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component configuration to a dictionary for saving or reconstruction."""
        return default_to_dict(self, audio_path=self.audio_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreWriterEnricher":
        """Creates a Dictator instance from a serialized dictionary of parameters."""
        return default_from_dict(cls, data)
