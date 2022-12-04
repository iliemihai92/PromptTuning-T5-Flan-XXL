import os
from pathlib import Path

from transformers import GPTJForCausalLM
import torch
import torch.nn as nn

device="cuda"

class GPTPromptTuningMixin:
    # SET DEVICE CUDA
    device = "cuda"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = False,
        random_range: float = 0.5,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs).cuda()

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device(device)
        )
        self.n_tokens = self.soft_prompt.num_embeddings

        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            print("Init from vocab...")
            init_prompt_value = self.transformer.wte.weight[:n_tokens].clone().detach()
        else:
            print("Init from random....")
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, 1536)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1).to("cuda")

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
 
        # put model on cuda
        self.transformer = self.transformer.to("cuda")

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )



class GPTJPromptTuningLM(GPTPromptTuningMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
