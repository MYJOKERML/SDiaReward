"""
FastWhisperFeatureExtractor - Optimized Whisper feature extractor

Key optimizations:
1. Batch pre-allocation of memory, avoiding per-sample np.pad
2. Uses PyTorch for batch padding (optional GPU acceleration)
3. Reduces dictionary operations and type conversion overhead

Interface is fully compatible with WhisperFeatureExtractor and can be used as a drop-in replacement.
"""

from typing import Optional, Union, List
import numpy as np
import torch

from transformers import WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType


class FastWhisperFeatureExtractor(WhisperFeatureExtractor):
    """
    Optimized WhisperFeatureExtractor, primarily optimizing padding performance.

    Usage:
        # Replace the original feature_extractor
        from utils.fast_whisper_feature_extractor import FastWhisperFeatureExtractor
        feature_extractor = FastWhisperFeatureExtractor.from_pretrained(model_path)
    """

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        return_token_timestamps: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Optimized feature extraction method with an interface identical to the original.
        """
        # Sampling rate check
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                f" was sampled with {self.sampling_rate} and not {sampling_rate}."
            )

        # Standardize input format
        is_batched = isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        if not is_batched:
            raw_speech = [raw_speech]

        # Convert to list of numpy arrays
        raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]

        # Determine max_length
        target_length = max_length if max_length else self.n_samples
        # Fast batch padding
        padded_waveforms, attention_mask = self._fast_batch_pad(
            raw_speech,
            target_length=target_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask or do_normalize,
        )

        # Zero-mean unit-variance normalization
        if do_normalize:
            padded_waveforms = self._fast_normalize(padded_waveforms, attention_mask)

        # Extract mel features
        extract_fbank_features = (
            self._torch_extract_fbank_features if torch.cuda.is_available() or device == "cpu"
            else self._np_extract_fbank_features
        )
        input_features = extract_fbank_features(padded_waveforms, device)

        # Build output
        result = {"input_features": input_features}

        if return_attention_mask or do_normalize:
            # Rescale from sample to feature (hop_length)
            rescaled_mask = attention_mask[:, ::self.hop_length]
            if attention_mask.shape[1] % self.hop_length != 0:
                rescaled_mask = rescaled_mask[:, :-1]
            result["attention_mask"] = rescaled_mask

        # Convert to specified tensor type
        if return_tensors == "pt":
            result = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in result.items()}
        elif return_tensors == "np":
            pass  # Already numpy

        return BatchFeature(result)

    def _fast_batch_pad(
        self,
        waveforms: List[np.ndarray],
        target_length: int,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: bool = True,
    ) -> tuple:
        """
        Fast batch padding with pre-allocated memory to avoid per-sample operations.

        Returns:
            padded_waveforms: [batch_size, target_length]
            attention_mask: [batch_size, target_length] or None
        """
        batch_size = len(waveforms)
        # Adjust target_length to be a multiple of pad_to_multiple_of
        if pad_to_multiple_of is not None and target_length % pad_to_multiple_of != 0:
            target_length = ((target_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        # Pre-allocate memory
        padded = np.full((batch_size, target_length), self.padding_value, dtype=np.float32)
        if return_attention_mask:
            attention_mask = np.zeros((batch_size, target_length), dtype=np.int32)
        else:
            attention_mask = None
        # Batch fill
        for i, wav in enumerate(waveforms):
            length = len(wav)

            if truncation and length > target_length:
                # Truncate
                padded[i] = wav[:target_length]
                if attention_mask is not None:
                    attention_mask[i] = 1
            else:
                # Pad (or exactly equal to target_length)
                actual_length = min(length, target_length)
                padded[i, :actual_length] = wav[:actual_length]
                if attention_mask is not None:
                    attention_mask[i, :actual_length] = 1

        return padded, attention_mask

    def _fast_normalize(
        self,
        waveforms: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Fast batch zero-mean unit-variance normalization.
        """
        if attention_mask is None:
            # No mask, directly batch normalize
            mean = waveforms.mean(axis=1, keepdims=True)
            std = np.sqrt(waveforms.var(axis=1, keepdims=True) + 1e-7)
            return (waveforms - mean) / std

        # With mask, need per-sample processing (but avoid creating new arrays)
        result = waveforms.copy()
        lengths = attention_mask.sum(axis=1)

        for i in range(len(waveforms)):
            length = lengths[i]
            if length > 0:
                valid_part = result[i, :length]
                mean = valid_part.mean()
                std = np.sqrt(valid_part.var() + 1e-7)
                result[i, :length] = (valid_part - mean) / std
                result[i, length:] = self.padding_value

        return result


class FastWhisperFeatureExtractorV2(WhisperFeatureExtractor):
    """
    More aggressive optimization version using PyTorch for all operations (including padding).
    Suitable for GPU scenarios where greater acceleration can be achieved.
    """

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        return_token_timestamps: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Full PyTorch implementation of feature extraction.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Sampling rate mismatch: expected {self.sampling_rate}, got {sampling_rate}"
            )

        # Standardize input
        is_batched = isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        if not is_batched:
            raw_speech = [raw_speech]

        target_length = max_length if max_length else self.n_samples
        if pad_to_multiple_of is not None and target_length % pad_to_multiple_of != 0:
            target_length = ((target_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # Convert to tensor and pad
        padded, attention_mask = self._torch_batch_pad(
            raw_speech, target_length, truncation, device
        )

        # Normalize
        if do_normalize:
            padded = self._torch_normalize(padded, attention_mask)

        # Extract mel features
        input_features = self._torch_extract_fbank_features_v2(padded, device)

        # Build output
        result = {"input_features": input_features}

        if return_attention_mask or do_normalize:
            rescaled_mask = attention_mask[:, ::self.hop_length]
            if attention_mask.shape[1] % self.hop_length != 0:
                rescaled_mask = rescaled_mask[:, :-1]
            result["attention_mask"] = rescaled_mask

        # Convert output format
        if return_tensors == "pt":
            pass  # Already torch tensor
        elif return_tensors == "np":
            result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        else:
            result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in result.items()}

        return BatchFeature(result)

    def _torch_batch_pad(
        self,
        waveforms: List,
        target_length: int,
        truncation: bool,
        device: str,
    ) -> tuple:
        """PyTorch batch padding"""
        batch_size = len(waveforms)

        # Pre-allocate tensor
        padded = torch.full(
            (batch_size, target_length),
            self.padding_value,
            dtype=torch.float32,
            device=device
        )
        attention_mask = torch.zeros(
            (batch_size, target_length),
            dtype=torch.int32,
            device=device
        )

        for i, wav in enumerate(waveforms):
            wav_tensor = torch.as_tensor(wav, dtype=torch.float32, device=device)
            length = len(wav_tensor)

            if truncation and length > target_length:
                padded[i] = wav_tensor[:target_length]
                attention_mask[i] = 1
            else:
                actual_length = min(length, target_length)
                padded[i, :actual_length] = wav_tensor[:actual_length]
                attention_mask[i, :actual_length] = 1

        return padded, attention_mask

    def _torch_normalize(
        self,
        waveforms: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch batch normalization"""
        lengths = attention_mask.sum(dim=1)
        result = waveforms.clone()

        for i in range(len(waveforms)):
            length = lengths[i].item()
            if length > 0:
                valid = result[i, :int(length)]
                mean = valid.mean()
                std = (valid.var() + 1e-7).sqrt()
                result[i, :int(length)] = (valid - mean) / std
                result[i, int(length):] = self.padding_value

        return result

    def _torch_extract_fbank_features_v2(
        self,
        waveform: torch.Tensor,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Optimized mel feature extraction, consistent with parent class
        _torch_extract_fbank_features but accepts already-padded input.
        """
        if waveform.device.type != device:
            waveform = waveform.to(device)

        window = torch.hann_window(self.n_fft, device=device)

        if self.dither != 0.0:
            waveform = waveform + self.dither * torch.randn_like(waveform)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.amax(dim=(1, 2), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec


def create_fast_feature_extractor(model_path: str, version: str = "v1") -> WhisperFeatureExtractor:
    """
    Convenience function to create a fast feature extractor.

    Args:
        model_path: Path to the model
        version: "v1" uses numpy-optimized version, "v2" uses full PyTorch version

    Returns:
        FastWhisperFeatureExtractor or FastWhisperFeatureExtractorV2 instance
    """
    if version == "v1":
        return FastWhisperFeatureExtractor.from_pretrained(model_path)
    elif version == "v2":
        return FastWhisperFeatureExtractorV2.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown version: {version}, expected 'v1' or 'v2'")
