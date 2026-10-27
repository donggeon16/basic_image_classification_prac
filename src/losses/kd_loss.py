import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, ce_loss, temperature=4.0, alpha=0.5):

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    return (1 - alpha) * ce_loss + alpha * kd_loss
