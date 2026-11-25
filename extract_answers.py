from utils import extract_letter
import re
import string


#######################
#   Multiple-Choice   #
#######################

def extract_answer_MC(text, options=None):
    if options:
        valid_letters = [chr(65+i) for i in range(len(options))]
    else:
        valid_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    #check if the start is "X}" 
    prefix_match = re.match(r'^\s*([A-Za-z])\}', text)
    if prefix_match:
        letter = prefix_match.group(1).upper()
        if letter in valid_letters:
            return letter

    #remove <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    #check for \answer{X} or answer{X}
    boxed_matches = re.findall(r'\\?answer\{([^}]*)\}', text, re.IGNORECASE)
    if boxed_matches:
        for content in reversed(boxed_matches):
            content_match = re.search(r'([A-Za-z])', content)
            if content_match:
                letter = content_match.group(1).upper()
                if letter in valid_letters:
                    return letter

    #check for "answer: X"
    answer_matches = re.findall(r'answer[:\s]+([A-Za-z])(?![A-Za-z])', text, re.IGNORECASE)
    if answer_matches:
        last_match = answer_matches[-1].upper()
        if last_match in valid_letters:
            return last_match

    return ""



#####################
#   Open-Question   #
#####################

def extract_answer_OQ(text, options=None):
    #check the sart and if '}' appears within the first 30 characters
    completion_match = re.match(r'^\s*([^}]{1,30})\}', text)
    
    if completion_match:
        completion_content = completion_match.group(1).strip()
        
        if re.search(r'[a-zA-Z0-9]', completion_content):
            if completion_content.lower() != "fridge":
                return completion_content

    
    #remove <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    #extract \answer{...}
    #boxed_matches = re.findall(r'\\?answer\{([^}]*)\}', text, re.IGNORECASE)
    boxed_matches = re.findall(r'\\?answer\{([^}]*)(?:\}|(?=$))', text, re.IGNORECASE)
    
    raw_answer = ""
    if boxed_matches:
        for match in reversed(boxed_matches):
            cleaned = match.strip().strip(string.punctuation + ' ')
            if cleaned and cleaned.lower() != "fridge":
                return cleaned

    #check for "Answer: [text]"
    patterns = [r'answer[:\s]+([^\n]+)', r'the answer is[:\s]+([^\n]+)']
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in reversed(matches):
                cleaned = match.strip().strip(string.punctuation + ' ')
                if cleaned and cleaned.lower() != "fridge":
                    return cleaned

    return ""







#########################
#   Memory extraction   #
#########################

def extract_memory(text):
    results = []

    #check \memory or memory
    pattern = re.compile(r'\\?memory\s*(\{)?', re.IGNORECASE)
    pos = 0

    while pos < len(text):
        match = pattern.search(text, pos)
        if not match:
            break

        has_brace = match.group(1) == '{'
        start = match.end()
        content = ""

        if has_brace:
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            content = text[start:i-1 if depth == 0 else len(text)].strip()
            pos = i
        else:
            #if no closing brace, extract until next memory or end of text
            next_match = pattern.search(text, start)
            end = next_match.start() if next_match else len(text)
            content = text[start:end].strip()
            pos = end

        results.append(content)

    return results[-1] if results else ""






###############################
#   Judge answer extrcation   #
###############################

def extract_answer_judge(text):
    #remove <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    #check " CORRECT}" -> "CORRECT"
    completion_match = re.match(r'^\s*([^}]{1,30})\}', text)
    if completion_match:
        content = completion_match.group(1).strip().upper()
        if "INCORRECT" in content: return "INCORRECT"
        if "CORRECT" in content: return "CORRECT"

    #check \answer{...}
    boxed_matches = re.findall(r'\\answer\{([^}]*)(?:\}|(?=$))', text, re.IGNORECASE)
    
    if boxed_matches:
        for match in reversed(boxed_matches):
            cleaned = match.strip().upper()
            if "INCORRECT" in cleaned:
                return "INCORRECT"
            if "CORRECT" in cleaned:
                return "CORRECT"

    #check "Answer: CORRECT" or "The answer is INCORRECT"
    patterns = [
        r'answer[:\s*]+(CORRECT|INCORRECT)', 
        r'the answer is[:\s*]+(CORRECT|INCORRECT)',
        #r'\b(CORRECT|INCORRECT)\b' # Last resort: just find the word anywhere
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip().upper()

    return ""



def extract_answer_judge_Custom(text):
    #remove <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    #check start" 42}" -> "42"
    completion_match = re.match(r'^\s*([+-]?\d?\.?\d+)\}', text)
    if completion_match:
        return completion_match.group(1).strip()

    #check \answer{...}
    boxed_matches = re.findall(r'\\answer\{([+-]?\d?\.?\d+)(?:\}|(?=$))', text, re.IGNORECASE)
    
    if boxed_matches:
        return boxed_matches[-1].strip()

    #check for "Answer: 42" or "The answer is 42"
    patterns = [
        r'answer[:\s*]+([+-]?\d?\.?\d+)', 
        r'the answer is[:\s*]+([+-]?\d?\.?\d+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    return ""