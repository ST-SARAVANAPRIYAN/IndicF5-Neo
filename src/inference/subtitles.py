"""Subtitle generation with CTC-first forced alignment and safe fallback."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


class SubtitleGenerator:
    """Generate SRT files from generated audio and known text."""

    LANGUAGE_MODEL_MAP = {
        "ta": "jonatasgrosman/wav2vec2-large-xlsr-53-tamil",
        "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
        "te": "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100",
        "kn": "Harveenchadha/vakyansh-wav2vec2-kannada-knm-56",
        "ml": "Harveenchadha/vakyansh-wav2vec2-malayalam-mlm-8",
        "bn": "arijitx/wav2vec2-xls-r-300m-bengali",
    }
    DEFAULT_MULTILINGUAL_MODEL = "facebook/wav2vec2-xls-r-300m"

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._processor = None
        self._model = None
        self._model_id = None

    def generate_srt(
        self,
        audio_path: str | Path,
        transcript: str,
        srt_path: str | Path | None = None,
        language: str = "ta",
    ) -> Tuple[str, str]:
        """Generate subtitle file and return `(srt_path, method_used)`."""
        text = (transcript or "").strip()
        if not text:
            raise ValueError("Transcript is empty; cannot generate subtitles")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        if srt_path is None:
            srt_path = audio_path.with_suffix(".srt")
        srt_path = Path(srt_path)

        method = "estimated"
        segments: List[SubtitleSegment]
        try:
            segments, method = self._align_ctc(audio_path=audio_path, transcript=text, language=language)
            logger.info("SRT alignment completed with CTC (%s)", method)
        except Exception as e:
            logger.warning("CTC alignment unavailable, using estimated timing: %s", str(e))
            segments = self._estimate_segments(audio_path=audio_path, transcript=text)

        self._write_srt(segments=segments, srt_path=srt_path)
        return str(srt_path), method

    def _align_ctc(self, audio_path: Path, transcript: str, language: str) -> Tuple[List[SubtitleSegment], str]:
        """Forced align transcript using a CTC acoustic model."""
        try:
            from ctc_segmentation import (
                CtcSegmentationParameters,
                ctc_segmentation,
                determine_utterance_segments,
                prepare_text,
            )
            from transformers import AutoModelForCTC, AutoProcessor
        except Exception as e:
            raise RuntimeError(
                "Missing CTC alignment deps. Install `ctc-segmentation` and ensure `transformers` is available"
            ) from e

        words = self._tokenize_words(transcript)
        if len(words) < 2:
            raise RuntimeError("Transcript too short for robust forced alignment")

        model_id = self._resolve_model_id(language)
        processor, model = self._load_ctc_model(model_id, AutoProcessor, AutoModelForCTC)

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        target_sr = processor.feature_extractor.sampling_rate
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr

        input_values = processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).input_values

        with torch.no_grad():
            logits = model(input_values.to(next(model.parameters()).device)).logits
        probs = torch.log_softmax(logits, dim=-1).cpu().numpy()[0]

        vocab = processor.tokenizer.get_vocab()
        index_to_token = {idx: token for token, idx in vocab.items()}
        char_list = [index_to_token[i] for i in range(len(index_to_token))]

        params = CtcSegmentationParameters(char_list=char_list)
        params.index_duration = float(waveform.shape[-1]) / float(sr) / float(probs.shape[0])

        ground_truth, utt_begin_indices = prepare_text(params, words)
        timings, char_probs, _ = ctc_segmentation(params, probs, ground_truth)
        word_segments = determine_utterance_segments(
            params,
            utt_begin_indices,
            char_probs,
            timings,
            words,
        )

        aligned_words: List[SubtitleSegment] = []
        prev_end = 0.0
        for entry in word_segments:
            text = str(entry[0]).strip()
            start = max(float(entry[1]), prev_end)
            end = max(float(entry[2]), start + 0.04)
            prev_end = end
            if text:
                aligned_words.append(SubtitleSegment(start=start, end=end, text=text))

        if not aligned_words:
            raise RuntimeError("CTC alignment returned empty segments")

        return self._group_words(aligned_words), f"ctc:{model_id}"

    def _load_ctc_model(self, model_id: str, auto_processor_cls, auto_model_cls):
        if self._processor is not None and self._model is not None and self._model_id == model_id:
            return self._processor, self._model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CTC alignment model: %s on %s", model_id, device)
        processor = auto_processor_cls.from_pretrained(model_id)
        model = auto_model_cls.from_pretrained(model_id).to(device)
        model.eval()
        self._processor = processor
        self._model = model
        self._model_id = model_id
        return processor, model

    def _resolve_model_id(self, language: str) -> str:
        lang = (language or "").strip().lower()
        env_override = os.getenv("INDICF5_CTC_ALIGN_MODEL", "").strip()
        if env_override:
            return env_override
        return self.LANGUAGE_MODEL_MAP.get(lang, self.DEFAULT_MULTILINGUAL_MODEL)

    def _tokenize_words(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        return [token for token in normalized.split(" ") if token]

    def _group_words(self, words: List[SubtitleSegment]) -> List[SubtitleSegment]:
        grouped: List[SubtitleSegment] = []
        current_words: List[str] = []
        start = words[0].start
        end = words[0].end

        for seg in words:
            candidate_words = current_words + [seg.text]
            candidate_text = " ".join(candidate_words).strip()
            candidate_duration = seg.end - start
            should_break = (
                len(candidate_text) > 42
                or candidate_duration > 5.0
                or (current_words and re.search(r"[.!?।]$", current_words[-1]))
            )

            if should_break and current_words:
                grouped.append(SubtitleSegment(start=start, end=end, text=" ".join(current_words).strip()))
                current_words = [seg.text]
                start = seg.start
                end = seg.end
            else:
                current_words.append(seg.text)
                end = seg.end

        if current_words:
            grouped.append(SubtitleSegment(start=start, end=end, text=" ".join(current_words).strip()))

        return grouped

    def _estimate_segments(self, audio_path: Path, transcript: str) -> List[SubtitleSegment]:
        """Fallback segmentation when forced alignment is unavailable."""
        try:
            duration = float(sf.info(str(audio_path)).duration)
        except Exception:
            duration = 0.0
        duration = max(duration, 0.1)

        text = re.sub(r"\s+", " ", transcript).strip()
        chunks = [c.strip() for c in re.split(r"(?<=[.!?।])\s+", text) if c.strip()]
        if not chunks:
            chunks = [text]

        weights = [max(len(chunk), 1) for chunk in chunks]
        total_weight = float(sum(weights))

        segments: List[SubtitleSegment] = []
        cursor = 0.0
        for chunk, weight in zip(chunks, weights):
            span = duration * (weight / total_weight)
            start = cursor
            end = min(duration, cursor + span)
            if end <= start:
                end = min(duration, start + 0.2)
            segments.append(SubtitleSegment(start=start, end=end, text=chunk))
            cursor = end

        if segments:
            segments[-1].end = duration
        return segments

    def _write_srt(self, segments: List[SubtitleSegment], srt_path: Path) -> None:
        lines: List[str] = []
        for idx, seg in enumerate(segments, start=1):
            lines.append(str(idx))
            lines.append(f"{self._fmt_ts(seg.start)} --> {self._fmt_ts(seg.end)}")
            lines.append(seg.text)
            lines.append("")

        srt_path.parent.mkdir(parents=True, exist_ok=True)
        srt_path.write_text("\n".join(lines), encoding="utf-8")

    def _fmt_ts(self, seconds: float) -> str:
        total_ms = int(max(seconds, 0.0) * 1000)
        hours = total_ms // 3_600_000
        minutes = (total_ms % 3_600_000) // 60_000
        secs = (total_ms % 60_000) // 1000
        ms = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
