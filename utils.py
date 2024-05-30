import re


def extract_answer(text_response, pattern=r'Answer: \[\/INST\]<<(?:SYS|INST)>> (.+?) \n') -> str:
    match = re.search(pattern, text_response)
    if match:
        answer = match.group(1).strip()
        return answer
    else:
        return 'Answer not found'
