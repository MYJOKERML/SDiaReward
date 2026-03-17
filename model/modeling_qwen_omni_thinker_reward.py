# qwen_omni_thinker_reward.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniConfig, Qwen2_5OmniThinkerConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.activations import ACT2FN
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, check_torch_load_is_safe, logging
from transformers.cache_utils import Cache


class LastTokenPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        # hidden_states: [B,T,H], mask: [B,T] (1=valid)
        idx = mask.long().sum(dim=1) - 1  # [B]
        idx = idx.clamp(min=0)
        b = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[b, idx], None


class MeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        # masked mean
        w = mask.unsqueeze(-1).to(hidden_states.dtype)               # [B,T,1]
        denom = w.sum(dim=1).clamp(min=1.0)                         # [B,1]
        pooled = (hidden_states * w).sum(dim=1) / denom             # [B,H]
        return pooled, None

class AttentionPooling(nn.Module):
    """Attention pooling for sentence-level representation"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, T, H], attention_mask: [B, T]
        attn_scores = self.attn(hidden_states).squeeze(-1)   # [B, T]
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)    # [B, T]
        pooled = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)  # [B, H]
        return pooled, attn_weights

def build_pooler(pooling_type: str, hidden_size: int) -> nn.Module:
    pooling_type = pooling_type.lower()
    if pooling_type in ["last", "last_token", "eos"]:
        return LastTokenPooling()
    if pooling_type in ["mean", "avg"]:
        return MeanPooling()
    if pooling_type in ["attn", "attention"]:
        return AttentionPooling(hidden_size)
    raise ValueError(f"Unknown pooling_type={pooling_type}")

class QwenOmniThinkerReward(Qwen2_5OmniThinkerForConditionalGeneration):
    """
    Reward Model version of Qwen2.5-Omni Thinker
    - Adds score_head for scalar/multi-dim reward
    - Uses attention pooling instead of last-token pooling
    """

    def __init__(self, config, num_rewards: int = 1, pooling_type: str = "mean"):
        """
        Args:
            config: Qwen config
            num_rewards: number of reward dimensions
                         (e.g., 1 = scalar reward)
        """
        super().__init__(config)
        self.num_rewards = num_rewards
        self.hidden_size = config.text_config.hidden_size

        self.pooling_type = pooling_type
        self.pooler = build_pooler(pooling_type, self.hidden_size)

        self.score_head = nn.Linear(self.hidden_size, num_rewards, bias=False)
        # init params
        self.post_init()

    def freeze_encoder(self, freeze_text=True):
        """Freeze encoder weights to stabilize RM training"""
        if hasattr(self, "audio_tower"):
            for p in self.audio_tower.parameters():
                p.requires_grad = False
            print('Audio tower frozen.')
        if hasattr(self, "visual"):
            for p in self.visual.parameters():
                p.requires_grad = False
            print('Visual module frozen.')
        if freeze_text:
            for p in self.model.parameters():
                p.requires_grad = False
            print('Text model frozen.')

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , image and video
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            return_audio=False,
            **kwargs,
        )

        hidden_states = outputs[0]   # [B, T, H]

        # pooling (attention)
        pooled, attn_weights = self.pooler(hidden_states, attention_mask)

        # reward prediction
        logits = self.score_head(pooled)   # [B, num_rewards]
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output
        if attention_mask is None:
            raise ValueError("attention_mask is required for pooling")

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
