import torch
import torch.nn.functional as F

class CalculatePerplexity:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_logprobs(self, sequences, input_len):
        """
        sequences -> output.sequences (tensor)
        input_len -> tamanho do prompt
        """

        with torch.no_grad():
            logits = self.model(sequences).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = sequences[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)

        token_logprobs = log_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # pegar só parte gerada
        generated_logprobs = token_logprobs[0][input_len-1:]
        generated_ids = sequences[0][input_len:]

        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)

        pairs = list(zip(tokens, generated_logprobs.tolist()))

        # métricas úteis
        mean_logprob = generated_logprobs.mean().item()
        perplexity = torch.exp(-generated_logprobs.mean()).item()

        return pairs, mean_logprob, perplexity
