def word_count(batch, count=None):
    if count is None:
        count = {}
    for text in batch:
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count
