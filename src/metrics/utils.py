def calc_cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return float(len(hyp))

    dp_table = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        dp_table[i][0] = i
    for j in range(len(hyp) + 1):
        dp_table[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp_table[i][j] = min(
                dp_table[i - 1][j] + 1,
                dp_table[i][j - 1] + 1,
                dp_table[i - 1][j - 1] + cost
            )

    edit_distance = dp_table[len(ref)][len(hyp)]
    return edit_distance / len(ref)


def calc_wer(target_text: str, predicted_text: str) -> float:
    ref_words = target_text.split()
    hyp_words = predicted_text.split()

    if len(ref_words) == 0:
        return float(len(hyp_words))

    ref_len = len(ref_words)
    hyp_len = len(hyp_words)
    dp_table = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    for i in range(ref_len + 1):
        dp_table[i][0] = i
    for j in range(hyp_len + 1):
        dp_table[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            dp_table[i][j] = min(
                dp_table[i - 1][j] + 1,
                dp_table[i][j - 1] + 1,
                dp_table[i - 1][j - 1] + cost
            )

    edit_distance = dp_table[ref_len][hyp_len]
    wer = edit_distance / ref_len
    return wer