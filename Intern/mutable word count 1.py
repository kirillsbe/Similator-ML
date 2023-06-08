import utils


def test_word_count():
    '''
    '''
    count = utils.word_count(['apple orange', 'banana apple'])
    assert count == {'apple': 2, 'orange': 1, 'banana': 1}


def test_word_count_tricky():
    count = utils.word_count(['apple orange', 'banana apple'])
    count = utils.word_count(['cherry pear', 'apple banana'])
    assert count == {'apple' : 1, 'banana' : 1, 'cherry' : 1, 'pear' : 1}