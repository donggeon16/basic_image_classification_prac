import torch.nn.functional as F
import torch


def distillation_loss(student_logits, teacher_logits, ce_loss, temperature=4.0, alpha=0.5):

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    return (1 - alpha) * ce_loss + alpha * kd_loss


def mixup_dkd_terms(student_logits, teacher_logits, labels, temperature):
    labels = labels.reshape(-1, 1)

    s = student_logits / temperature
    t = teacher_logits.detach() / temperature

    target_mask = torch.zeros_like(s, dtype=torch.bool)
    target_mask.scatter_(1, labels, True)

    s_two = torch.cat([s.gather(1, labels), torch.logsumexp(s.masked_fill(target_mask, -torch.inf), dim=1, keepdim=True)], dim=1)

    t_two = torch.cat([
        t.gather(1, labels),
        torch.logsumexp(t.masked_fill(target_mask, -torch.inf), dim=1, keepdim=True)], dim=1)

    tckd = F.kl_div(F.log_softmax(s_two, dim=1), F.softmax(t_two, dim=1), reduction="batchmean") * (temperature ** 2)

    s_other = s.masked_fill(target_mask, -1e4)
    t_other = t.masked_fill(target_mask, -1e4)

    nckd = F.kl_div(F.log_softmax(s_other, dim=1), F.softmax(t_other, dim=1), reduction="batchmean") * (temperature ** 2)

    return tckd, nckd

def dkd_loss(
    student_logits,
    teacher_logits,
    labels,
    ce_loss,
    temperature=4.0,
    alpha=1.0,
    beta=8.0,
    labels_b=None,
    lam=1.0,
    warmup_factor=1.0
):
    tckd, nckd = mixup_dkd_terms(student_logits, teacher_logits, labels, temperature)

    if labels_b is not None:
        tckd_b, nckd_b = mixup_dkd_terms(student_logits, teacher_logits, labels_b, temperature)

        tckd = lam * tckd + (1 - lam) * tckd_b
        nckd = lam * nckd + (1 - lam) * nckd_b

    return ce_loss + warmup_factor * (alpha * tckd + beta * nckd)