import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from model import VLM


class FakeVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        cls_token = torch.tensor([1.0, 2.0, 3.0, 4.0])
        patch_token = torch.tensor([9.0, 9.0, 9.0, 9.0])
        hidden_state = torch.stack((cls_token, patch_token))
        hidden_state = hidden_state.expand(batch_size, -1, -1).clone()
        return type("Output", (), {"last_hidden_state": hidden_state})()


class FakeVisionInputs(dict):
    def to(self, device):
        return FakeVisionInputs(
            {name: value.to(device) for name, value in self.items()}
        )


class FakeVisionProcessor:
    def __call__(self, images, return_tensors):
        return FakeVisionInputs(
            pixel_values=torch.zeros(len(images), 3, 2, 2)
        )


class CaptureLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(4, 4)
        self.input_tensor = None
        self.attention_mask = None

    def forward(self, tensor, causal_mask, fine_tuning):
        self.input_tensor = tensor.detach().clone()
        self.attention_mask = causal_mask.detach().clone()
        return tensor, torch.tensor(0.0)


class VLMTest(unittest.TestCase):
    def test_forward_uses_clip_cls_token_and_supports_text_only_samples(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            embeddings_path = temporary_path / "embeddings.pt"
            image_path = temporary_path / "image.png"
            torch.save(torch.randn(8, 4), embeddings_path)
            Image.new("RGB", (2, 2), color="red").save(image_path)

            with (
                patch(
                    "model.CLIPVisionModel.from_pretrained",
                    return_value=FakeVisionModel(),
                ),
                patch(
                    "model.CLIPImageProcessor.from_pretrained",
                    return_value=FakeVisionProcessor(),
                ),
            ):
                model = VLM(
                    num_layer=1,
                    max_context_length=3,
                    word_embeddings_tensor=str(embeddings_path),
                    projection_dim=4,
                    expansion_factor=2,
                    head_dim=2,
                    q_head=2,
                    kv_head=1,
                    device="cpu",
                )

            capture_llm = CaptureLLM()
            model.llm = capture_llm
            token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
            prediction, load_balancing_loss = model(
                token_ids,
                [str(image_path), None],
            )

            self.assertEqual(prediction.shape, (2, 3, 4))
            self.assertEqual(load_balancing_loss.item(), 0.0)
            self.assertTrue(
                torch.equal(
                    capture_llm.input_tensor[0, 0],
                    torch.tensor([1.0, 2.0, 3.0, 4.0]),
                )
            )
            self.assertTrue(
                torch.isneginf(
                    capture_llm.attention_mask[1, 0, 2:, :2]
                ).all()
            )


if __name__ == "__main__":
    unittest.main()
