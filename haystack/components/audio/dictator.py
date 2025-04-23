import gc
import glob
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

import torch
from dotenv import load_dotenv

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports
with LazyImport("Run 'pip install whisperx' to use Dictator.") as whisperx_import:
    import whisperx
with LazyImport("Run 'pip install pyannote.audio' to use speaker diarization.") as pyannote_import:
    from pyannote.audio import Pipeline as PyAnnotePipeline
with LazyImport("Run 'pip install soundfile' to handle audio file I/O.") as soundfile_import:
    pass


def load_project_env():
    """Load project env"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    logger.info(f"Loading env from: {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)


class DiarizationSegment(TypedDict):
    speaker: str
    start: float
    end: float


class WhisperSegment(TypedDict):
    speaker: str
    start: float
    end: float
    text: str


class FormattedSegment(TypedDict):
    speaker: str
    start_time: float
    end_time: float
    text: str


@component
class Dictator:
    def __init__(self, **kwargs):
        """Initialize Dictator"""
        whisperx_import.check()
        pyannote_import.check()
        soundfile_import.check()

        self.model_name = kwargs.get("model_name", "tiny")
        self.hf_token = kwargs.get("hf_token")
        self.min_speakers = kwargs.get("min_speakers", 2)
        self.max_speakers = kwargs.get("max_speakers", 6)
        self.batch_size = kwargs.get("batch_size", 16)
        self.compute_type = kwargs.get("compute_type", "float16")
        self.language = kwargs.get("language", "en")
        self.min_audio_length = kwargs.get("min_audio_length", 10.0)
        self.chunk_size = kwargs.get("chunk_size", 8.0)
        self.auto_upgrade_models = kwargs.get("auto_upgrade_models", False)
        self.use_direct_pyannote = kwargs.get("use_direct_pyannote", True)
        self.model: Optional[Any] = None

        self.device = self._setup_cuda()

        valid_models = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
        if self.model_name not in valid_models:
            logger.warning("'%s' not recognized. Defaulting to 'tiny'.", self.model_name)
            self.model_name = "tiny"

        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "int8"
            logger.info("Switching compute_type to int8 for CPU processing.")

        if self.auto_upgrade_models:
            self._upgrade_pytorch_checkpoints()

    def _setup_cuda(self) -> str:
        """Setup cuda"""
        logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
        logger.info("torch version: %s", torch.__version__)
        logger.info("torch version: %s", torch.__version__)
        if torch.cuda.is_available():
            try:
                if not torch.version.cuda or "+cu" not in torch.__version__:
                    print("⚠️ Detected CPU-only torch. Installing GPU-enabled version (cu121)...")
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--upgrade",
                            "torch==2.5.1+cu121",
                            "torchvision==0.20.1+cu121",
                            "torchaudio==2.5.1+cu121",
                            "--index-url",
                            "https://download.pytorch.org/whl/cu121",
                        ]
                    )
                    importlib.reload(torch)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_device(0)
                logger.info("🧠 Using GPU: %s", torch.cuda.get_device_name(0))
                return "cuda"
            except Exception as e:
                logger.warning("🚨 Torch GPU setup failed: %s", e)
        logger.info("🧠 CUDA not available or fallback triggered. Using CPU.")
        return "cpu"

    def _lazy_import_or_install(self, module: str, pip_name: Optional[str] = None):
        """Lazy import function for additional modules"""
        try:
            return __import__(module)
        except ImportError:
            pip_name = pip_name or module
            print(f"📦 Installing missing dependency: {pip_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return __import__(module)

    def _upgrade_pytorch_checkpoints(self):
        """Upgrades pytorch checkpoints"""
        whisperx_dir = os.path.dirname(whisperx.__file__)
        assets_dir = os.path.join(whisperx_dir, "assets")
        for model_file in glob.glob(os.path.join(assets_dir, "*.bin")):
            backup = f"{model_file}.backup"
            if not os.path.exists(backup):
                shutil.copy2(model_file, backup)
                subprocess.run(
                    [sys.executable, "-m", "pytorch_lightning.utilities.upgrade_checkpoint", model_file], check=False
                )

    def _format_timestamp(self, seconds: float) -> str:
        """
        Convert a time duration in seconds to SRT timestamp format (HH:MM:SS,mmm).

        Args:
            seconds (float): The time duration in seconds.

        Returns:
            str: The formatted timestamp string.
        """
        td = timedelta(seconds=seconds)
        total = int(td.total_seconds())
        ms = int((td.total_seconds() - total) * 1000)
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    def _generate_srt(self, segments: List[FormattedSegment]) -> str:
        """
        Generate a string in SRT subtitle format from a list of formatted segments.

        Args:
            segments (List[FormattedSegment]): List of speaker-labeled transcription segments.

        Returns:
            str: SubRip subtitle (SRT) formatted string.
        """
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(
                f"{i}\n"
                f"{self._format_timestamp(seg['start_time'])} --> {self._format_timestamp(seg['end_time'])}\n"
                f"{seg['speaker']}: {seg['text']}\n"
            )
        return "\n".join(lines)

    def warm_up(self):
        """Load the WhisperX model into memory."""
        whisperx_import.check()
        self.model = whisperx.load_model(
            self.model_name, self.device, compute_type=self.compute_type, language=self.language
        )

    @component.output_types(documents=List[Document])
    def run(self, input_file: str) -> Dict[str, Any]:
        """Transcribes and diarizes the provided audio file."""
        whisperx_import.check()
        p = Path(input_file)
        if not p.exists():
            raise FileNotFoundError(input_file)
        audio = whisperx.load_audio(str(p))
        length = len(audio) / 16000
        if not self.model:
            self.warm_up()
        if self.model is None:
            raise RuntimeError("Model not loaded. Call warm_up() first.")
        res = self.model.transcribe(  # mypy now knows this is safe
            audio=audio, batch_size=self.batch_size, chunk_size=self.chunk_size
        )

        # Align segments to get word-level timing
        logger.info("Aligning transcription with whisperx...")
        lang = res.get("language", self.language) or "en"
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
        segments = res.get("segments", [])
        aligned = whisperx.align(segments, align_model, metadata, audio, self.device, return_char_alignments=False)
        if "segments" in aligned:
            res = aligned

        segments = res.get("segments", [])
        if self.hf_token and len(segments) > 0 and length >= self.min_audio_length:
            logger.info("Running PyAnnote diarization...")
            diarization = self._process_diarization(str(p), audio)
            segments = self._assign_speakers_to_segments(segments, diarization)
            res["segments"] = segments

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        fmt = []
        for sg in segments:
            fmt.append(
                {
                    "speaker": sg.get("speaker", "Unknown"),
                    "start_time": float(sg["start"]),
                    "end_time": float(sg["end"]),
                    "text": sg["text"],
                }
            )

        idx, turns = [], []
        spmap: Dict[str, List[int]] = {}
        c = 0
        wid = 0

        for tid, sg in enumerate(fmt):
            words = sg["text"].split()
            ts = c
            wids = []
            for w in words:
                ce = c + len(w)
                idx.append(
                    {
                        "word_id": wid,
                        "word": w,
                        "char_start": c,
                        "char_end": ce,
                        "start_time": sg["start_time"],
                        "end_time": sg["end_time"],
                    }
                )
                wids.append(wid)
                wid += 1
                c = ce + 1
            turns.append(
                {
                    "turn_id": tid,
                    "speaker": sg["speaker"],
                    "start_time": sg["start_time"],
                    "end_time": sg["end_time"],
                    "char_start": ts,
                    "char_end": c - 1,
                    "word_ids": wids,
                }
            )
            key = re.sub(r"\W|^(?=\d)", "_", sg["speaker"])
            spmap.setdefault(key, []).append(tid)

        txt = " ".join(w["word"] for w in idx)
        doc = Document(
            content=txt,
            meta={
                "file_name": p.name,
                "file_path": str(p),
                "language": lang,
                "model_used": self.model_name,
                "speakers_detected": len({s["speaker"] for s in fmt}),
                "duration": fmt[-1]["end_time"] if fmt else 0,
                "segments": fmt,
                "words": idx,
                "turns": turns,
                "speakers": spmap,
                "content_type": "audio_transcript",
                "has_summary": False,
            },
        )
        return {"documents": [doc]}

    def _load_diarization_model(self):
        """Loads diarization model"""
        pyannote_import.check()
        versions = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization-3.0",
            "pyannote/speaker-diarization@2.1",
        ]
        for v in versions:
            try:
                mdl = PyAnnotePipeline.from_pretrained(v, use_auth_token=self.hf_token)
                if self.device == "cuda":
                    mdl.to(torch.device("cuda"))
                return mdl
            except:
                continue
        return PyAnnotePipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=self.hf_token)

    def _process_diarization(self, path: str, data, sr: int = 16000) -> List[DiarizationSegment]:
        """Processes diarization off the audio, makes a temp .wav to work with pyannote."""
        pyannote_import.check()
        mdl = self._load_diarization_model()
        sf = self._lazy_import_or_install("soundfile")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = tf.name
            sf.write(tmp, data, sr)

        dia = mdl(tmp, min_speakers=self.min_speakers, max_speakers=self.max_speakers)
        segments: List[Dict[str, Any]] = []

        for seg, _, sp in dia.itertracks(yield_label=True):
            segments.append({"speaker": f"Speaker {sp.split('_')[-1]}", "start": seg.start, "end": seg.end})

        os.unlink(tmp)
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return cast(List[DiarizationSegment], sorted(segments, key=lambda x: x["start"]))

    def _assign_speakers_to_segments(
        self, segments: List[Dict[str, Any]], diarization: List[DiarizationSegment]
    ) -> List[Dict[str, Any]]:
        """
        Assigns speakers to transcript segments based on diarization overlaps.

        Falls back to closest match if no direct overlap is found.
        """
        if not diarization:
            logger.warning("No diarization output found.")
            for i, seg in enumerate(segments):
                seg["speaker"] = f"Speaker {i % self.min_speakers}"
            return segments

        for seg in segments:
            overlaps: Dict[str, float] = {}
            seg_start = seg["start"]
            seg_end = seg["end"]
            mid = (seg_start + seg_end) / 2

            for d in diarization:
                overlap_start = max(seg_start, d["start"])
                overlap_end = min(seg_end, d["end"])
                if overlap_start < overlap_end:
                    overlaps[d["speaker"]] = overlaps.get(d["speaker"], 0) + (overlap_end - overlap_start)

            seg["speaker"] = (
                max(overlaps.items(), key=lambda x: x[1])[0]
                if overlaps
                else min(diarization, key=lambda d: abs(mid - ((d["start"] + d["end"]) / 2)))["speaker"]
            )
        return segments

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component configuration to a dictionary for saving or reconstruction."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            hf_token=self.hf_token,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            batch_size=self.batch_size,
            compute_type=self.compute_type,
            language=self.language,
            min_audio_length=self.min_audio_length,
            chunk_size=self.chunk_size,
            auto_upgrade_models=self.auto_upgrade_models,
            use_direct_pyannote=self.use_direct_pyannote,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dictator":
        """Creates a Dictator instance from a serialized dictionary of parameters."""
        return default_from_dict(cls, data)
