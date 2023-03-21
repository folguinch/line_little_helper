"""A variety of useful tools."""
def normalize_qns(qns: str) -> str:
    """Replace QNs characters for file naming."""
    new_qns = qns + ''
    to_replace = ['(', ')', ',', '/', 'J=', '=', '++', '--']
    for char in to_replace:
        if char == '++':
            new_qns = new_qns.replace(char, 'pp')
        elif char == '--':
            new_qns = new_qns.replace(char, 'mm')
        else:
            new_qns = new_qns.replace(char, '_')

    return new_qns
