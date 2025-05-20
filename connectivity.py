import torch
import torch.nn.functional as F
import numpy as np

def collect_connectivity(model, src_tensor, tgt_tensor,
                         src_idx2char, tgt_idx2char,
                         sos_idx, device="cuda"):
    model.eval()
    src_tensor = src_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out, enc_hidden = model.encoder(src_tensor, [src_tensor.size(1)])

    y_len = tgt_tensor.size(0)
    influence = torch.zeros(y_len, src_tensor.size(1), device=device)
    dec_hidden = enc_hidden
    prev = torch.tensor([sos_idx], device=device)

    for t in range(y_len):
        if hasattr(model.decoder, 'attention'):
            logits, dec_hidden, _ = model.decoder(prev, dec_hidden, enc_out)
        else:
            logits, dec_hidden = model.decoder(prev, dec_hidden)

        log_p = F.log_softmax(logits, dim=-1)[0, tgt_tensor[t]]
        model.zero_grad()
        log_p.backward(retain_graph=True)
        grad = enc_out.grad.detach()
        influence[t] = grad.norm(dim=-1)
        enc_out.grad.zero_()
        prev = tgt_tensor[t].unsqueeze(0)

    influence -= influence.min()
    influence /= influence.max().clamp(1e-6)

    return (influence.cpu().numpy(),
            [src_idx2char[i.item()] for i in src_tensor[0]],
            [tgt_idx2char[i.item()] for i in tgt_tensor])
