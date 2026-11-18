from datasets import load_dataset
from typing import Iterable, Dict


########    NORMAL
SYSTEM_MESSAGE_MC = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to select the correct option based on the provided narrative.\n"
    "RULES:\n"
    "1. FORMAT: Output ONLY the letter of the chosen option inside \\answer{{}}.\n"
    "2. CONSTRAINT: Do not provide reasoning, explanations, or any text outside the tag.\n"
    "EXAMPLE: If the correct option is 'A. The kitchen', your output must be \\answer{{A}}.\n"
)
FOOTER_MC = "You must output ONLY the chosen option letter inside \\answer{{}}."



SYSTEM_MESSAGE_OQ = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to provide the correct answer based on the provided narrative.\n"
    "RULES:\n"
    "1. FORMAT: Output ONLY the answer, ONE WORD, inside \\answer{{}}.\n"
    "2. CONSTRAINT: Do not provide reasoning, explanations, or any text outside the tag.\n"
    "EXAMPLE: If the correct answer is 'the fridge', your output must be \\answer{{fridge}}.\n"
)
FOOTER_OQ = "You must output ONLY the answer, ONE WORD, inside \\answer{{}}."







######  CoT
SYSTEM_MESSAGE_MC_CoT = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to reason step by step and select the correct option based on the provided narrative.\n"
    "RULES:\n"
    "1. FORMAT: Output the letter of the chosen option inside \\answer{{}}.\n"
    "2. CONSTRAINT: Provide your step-by-step reasoning before the tag.\n"
    "EXAMPLE: [Step-by-step reasoning...] Therefore, the answer is \\answer{{A}}.\n"
)
FOOTER_MC_CoT = "Reason step by step and output the chosen option letter inside \\answer{{}}."



SYSTEM_MESSAGE_OQ_CoT = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to reason step by step and provide the correct answer based on the provided narrative.\n"
    "RULES:\n"
    "1. FORMAT: Output the answer, ONE WORD, inside \\answer{{}}.\n"
    "2. CONSTRAINT: Provide your step-by-step reasoning before the tag.\n"
    "EXAMPLE: [Step-by-step reasoning...] Therefore, the answer is \\answer{{fridge}}.\n"
)
FOOTER_OQ_CoT = "Reason step by step and output the answer, ONE WORD, inside \\answer{{}}."







####### COMPLETE
SYSTEM_MESSAGE_MC_COMPLETE = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to track the mental states of all characters throughout the narrative and select the correct option.\n"
    "RULES:\n"
    "1. INITIAL STATE: Start by listing what each person knows and believes at the beginning of the narrative.\n"
    "2. PERSPECTIVE UPDATES: Describe how each person's beliefs change (or stay the same) as events occur.\n"
    "3. FORMAT: After this analysis, output the letter of the chosen option inside \\answer{{}}.\n"
    "4. CONSTRAINT: The belief tracking and updates must come BEFORE the final answer tag.\n"
    "EXAMPLE: [Initial beliefs... Updates based on events...] Therefore, the answer is \\answer{{A}}.\n"
)

FOOTER_MC_COMPLETE = "Track the mental states step by step and output the chosen option letter inside \\answer{{}}."


SYSTEM_MESSAGE_OQ_COMPLETE = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to track the mental states of all characters throughout the narrative and provide the correct answer.\n"
    "RULES:\n"
    "1. INITIAL STATE: Start by listing what each person knows and believes at the beginning of the narrative.\n"
    "2. PERSPECTIVE UPDATES: Describe how each person's beliefs change (or stay the same) as events occur.\n"
    "3. FORMAT: After this analysis, output the answer, ONE WORD, inside \\answer{{}}.\n"
    "4. CONSTRAINT: The belief tracking and updates must come BEFORE the final answer tag.\n"
    "EXAMPLE: [Initial beliefs... Updates based on events...] Therefore, the answer is \\answer{{fridge}}.\n"
)

FOOTER_OQ_COMPLETE = "Track the mental states step by step and output the answer, ONE WORD, inside \\answer{{}}."







######  EbE STANDARD

'''
#old/original version
SYSTEM_MESSAGE_EbE = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to update the 'Past Memory' based on the 'New Event' and return the new complete, updated memory.\n"
    "RULES:\n"
    "1. STATE OVERWRITE: For every entity (person or object), maintain only their current, most recent state. If a 'New Event' changes an entity's location or belief, delete the old fact and replace it with the new one.\n"
    "2. SCOPE: Track the physical locations of objects/people and the internal mental states (beliefs, knowledge, intentions) of all characters.\n"
    "3. SUMMARY: Do not transcribe the event text in the memory. Extract and store only the essential state changes.\n"
    "4. FORMAT: Return ONLY the updated memory inside \\memory{{}}.\n"
    "EXAMPLE: \\memory{{The ball is in the box; Anne is in the room and believes the ball is in the box; Bob is in the garden and believes the ball is in the basket.}}.\n"
)'''

#new version to try (improved)
SYSTEM_MESSAGE_EbE = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to update the 'Past Memory' based on the 'New Event' and return the new complete, updated memory, keeping track of the current locations and beliefs of all objects and people.\n"
    "RULES:\n"
    "1. SCOPE: Track the current physical locations of objects and people, and the internal mental states (beliefs, knowledge, intentions) of all characters.\n"
    "2. STATE OVERWRITE: For every entity (person or object), maintain only their current, most recent belief and location. If a 'New Event' changes an entity's location, delete the old fact and replace it with the new one.\n"
    "3. SUMMARY: Do not transcribe the event text in the memory. Extract and store only the essential subjects and relative locations.\n"
    "4. FORMAT: Return ONLY the updated memory inside \\memory{{}}.\n"
    "EXAMPLE: \\memory{{The ball is in the box; Anne is in the room and believes the ball is in the box; Bob is in the garden and believes the ball is in the basket.}}.\n"
)

FOOTER_EbE = "Output the updated memory inside \\memory{{}}."



SYSTEM_MESSAGE_MC_EbE_FINAL = (
    "You are a helpful assistant specialized in Theory of Mind."
    "Your task is to select the correct option based on the provided memory.\n"
    "RULES:\n"
    "1. FORMAT: Output ONLY the letter of the chosen option inside \\answer{{}}.\n"
    "EXAMPLE: If the correct option is 'A. The kitchen', your output must be \\answer{{A}}.\n"
)

FOOTER_MC_EbE_FINAL = "You must output ONLY the chosen option letter inside \\answer{{}}."


SYSTEM_MESSAGE_OQ_EbE_FINAL = (
    "You are a helpful assistant specialized in Theory of Mind."
    "Your task is to provide the correct answer based on the provided memory.\n"
    "RULES:\n"
    "1. FORMAT: Output ONLY the answer, ONE WORD, inside \\answer{{}}.\n"
    "EXAMPLE: If the correct answer is 'the fridge', your output must be \\answer{{fridge}}.\n"
)

FOOTER_OQ_EbE_FINAL = "You must output ONLY the answer, ONE WORD, inside \\answer{{}}."



######  EbE LOCATIONS

SYSTEM_MESSAGE_EbE_LOCATIONS = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to update the 'Past Memory' based on the 'New Event' and return the new complete, updated memory, keeping track ONLY of the current locations of all objects and people.\n"
    "RULES:\n"
    "1. SCOPE: Track ONLY the current physical locations of all objects and people.\n"
    "2. STATE OVERWRITE: For every entity (person or object), maintain only their current, most recent location. If a 'New Event' changes an entity's location, delete the old fact and replace it with the new one.\n"
    "3. SUMMARY: Do not transcribe the event text in the memory. Extract and store only the essential subjects and relative locations.\n"
    "4. FORMAT: Return ONLY the updated memory inside \\memory{{}}.\n"
    "EXAMPLE: \\memory{{The ball is in the box; Anne is in the room; Bob is in the garden.}}.\n"
)

######  EbE LOCATIONS OBJECTS

SYSTEM_MESSAGE_EbE_OBJ_LOCATIONS = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to update the 'Past Memory' based on the 'New Event' and return the new complete, updated memory, keeping track ONLY of the current locations of all objects.\n"
    "RULES:\n"
    "1. SCOPE: Track ONLY the current physical locations of all objects. Ignore people.\n"
    "2. STATE OVERWRITE: For every object, maintain only their current, most recent location. If a 'New Event' changes an object's location, update the memory deleting the old location and replacing it with the new one.\n"
    "3. SUMMARY: Do not transcribe the event text in the memory. Extract and store only the essential subjects and relative locations.\n"
    "4. FORMAT: Return ONLY the updated memory inside \\memory{{}}.\n"
    "EXAMPLE: \\memory{{The ball is in the box; The apple is in the fridge.}}.\n"
)


######  EbE BELIEFS

SYSTEM_MESSAGE_EbE_BELIEFS = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to update the 'Past Memory' based on the 'New Event' and return the new complete and updated memory, keeping track ONLY of the locations and beliefs of all people.\n"
    "RULES:\n"
    "1. SCOPE: Track ONLY the current locations and beliefs of all people.\n"
    "2. STATE OVERWRITE: For every person, maintain only their current location and beliefs. If a 'New Event' changes an entity's location or belief, update the memory deleting the old fact and replacing it with the new one.\n"
    "3. SUMMARY: Do not transcribe the event text in the memory. Extract and store only the essential subjects and relative beliefs.\n"
    "4. FORMAT: Return ONLY the updated memory inside \\memory{{}}.\n"
    "EXAMPLE: \\memory{{Anne is in the room and believes the ball is in the box; Bob is in the garden and believes the ball is in the basket.}}.\n"
)



####### TASK SPECIFIC EbE


SYSTEM_MESSAGE_EbE_TASK_SPECIFIC = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to update the 'Past Memory' based on the 'New Event' and return the new complete, updated memory, keeping track of the current locations and beliefs of all objects and people.\n"
    "RULES:\n"
    "1. SCOPE: Track the current physical locations of objects and people, and the internal mental states (beliefs, knowledge, intentions) of all characters.\n"
    "2. STATE OVERWRITE: For every entity (person or object), maintain only their current, most recent belief and location. If a 'New Event' changes an entity's location, delete the old fact and replace it with the new one.\n"
    "3. SUMMARY: Do not transcribe the event text in the memory. Extract and store only the essential subjects and relative locations.\n"
    "4. FORMAT: Return ONLY the updated memory inside \\memory{{}}.\n"
    "5. NESTED BELIEFS: Track not only what a person believes about an object, but also what they believe about other people's knowledge (e.g., 'X believes that Y thinks Z').\n"
    "EXAMPLE: \\memory{{The ball is in the box; Anne is in the room and believes the ball is in the box; Bob is in the garden and believes the ball is in the basket.}}.\n"
)








##############
#   BigToM   #
##############

def build_prompt_BigToM(example, shuffled_answers=None):
    narrative = " ".join(example["narrative"])
    question = example["question"]
    if shuffled_answers:
        options = f"\nOptions:\nA. {shuffled_answers[0]}\nB. {shuffled_answers[1]}"
        question = f"{question}{options}"

    user_content = (
        f"Story: {narrative}\n"
        f"Question: {question}\n"
        f"{FOOTER_MC}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_MC},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_BigToM_CoT(example, shuffled_answers=None):
    narrative = " ".join(example["narrative"])
    question = example["question"]
    if shuffled_answers:
        options = f"\nOptions:\nA. {shuffled_answers[0]}\nB. {shuffled_answers[1]}"
        question = f"{question}{options}"


    user_content = (
        f"Story: {narrative}\n"
        f"Question: {question}\n"
        f"{FOOTER_MC_CoT}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_MC_CoT},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_BigToM_Complete(example, shuffled_answers=None):
    narrative = " ".join(example["narrative"])
    question = example["question"]
    if shuffled_answers:
        options = f"\nOptions:\nA. {shuffled_answers[0]}\nB. {shuffled_answers[1]}"
        question = f"{question}{options}"


    user_content = (
        f"Story: {narrative}\n"
        f"Question: {question}\n"
        f"{FOOTER_MC_COMPLETE}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_MC_COMPLETE},
        {"role": "user", "content": user_content}
    ]

    return message





def build_prompt_BigToM_EbE(event, memory, question=None, shuffled_answers=None, ebe_type="standard"):
    if question:
        system_message = SYSTEM_MESSAGE_MC_EbE_FINAL

        options = f"\nOptions:\nA. {shuffled_answers[0]}\nB. {shuffled_answers[1]}"
        question = f"{question}{options}"

        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"{FOOTER_MC_EbE_FINAL}"
        )

    else:
        if ebe_type == "obj_location":
            system_message = SYSTEM_MESSAGE_EbE_OBJ_LOCATIONS
        elif ebe_type == "location":
            system_message = SYSTEM_MESSAGE_EbE_LOCATIONS
        elif ebe_type == "belief":
            system_message = SYSTEM_MESSAGE_EbE_BELIEFS
        elif ebe_type == "task_specific":
            system_message = SYSTEM_MESSAGE_EbE_TASK_SPECIFIC
        else:
            system_message = SYSTEM_MESSAGE_EbE

        user_content = (
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_EbE}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message




#############
#   HiToM   #
#############

def build_prompt_HiToM(example):
    #unify the story (merge the phrases)
    story = " ".join(example["story"])
    question = example.get("question", "")
    options = example.get("options", None)

    option_lines = "\n".join(options)

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"Options: {option_lines}\n"
        f"{FOOTER_MC}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_MC},
        {"role": "user", "content": user_content}
    ]
    

    return message



def build_prompt_HiToM_CoT(example):
    #unify the story (merge the phrases)
    story = " ".join(example["story"])
    question = example.get("question", "")
    options = example.get("options", None)

    option_lines = "\n".join(options)

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"Options: {option_lines}\n"
        f"{FOOTER_MC_CoT}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_MC_CoT},
        {"role": "user", "content": user_content}
    ]

    return message



def build_prompt_HiToM_Complete(example):
    #unify the story (merge the phrases)
    story = " ".join(example["story"])
    question = example.get("question", "")
    options = example.get("options", None)

    option_lines = "\n".join(options)

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"Options: {option_lines}\n"
        f"{FOOTER_MC_COMPLETE}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_MC_COMPLETE},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_HiToM_EbE(event, memory, question=None, options=None, ebe_type="standard"):
    if question:
        system_message = SYSTEM_MESSAGE_MC_EbE_FINAL

        option_lines = "\n".join(options)
        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"Options: {option_lines}\n"
            f"{FOOTER_MC_EbE_FINAL}"
        )

    else:
        if ebe_type == "obj_location":
            system_message = SYSTEM_MESSAGE_EbE_OBJ_LOCATIONS
        elif ebe_type == "location":
            system_message = SYSTEM_MESSAGE_EbE_LOCATIONS
        elif ebe_type == "belief":
            system_message = SYSTEM_MESSAGE_EbE_BELIEFS
        elif ebe_type == "task_specific":
            system_message = SYSTEM_MESSAGE_EbE_TASK_SPECIFIC
        else:
            system_message = SYSTEM_MESSAGE_EbE

        user_content = (
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_EbE}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message
    




#####################
#   ToMChallenges   #
#####################

def build_prompt_ToMChallenges(example):
    story = " ".join(example["story"])
    question = example["question"]

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"{FOOTER_OQ}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_OQ},
        {"role": "user", "content": user_content}
    ]
    
    return message


def build_prompt_ToMChallenges_CoT(example):
    story = " ".join(example["story"])
    question = example["question"]

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"{FOOTER_OQ_CoT}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_OQ_CoT},
        {"role": "user", "content": user_content}
    ]

    return message

def build_prompt_ToMChallenges_Complete(example):
    story = " ".join(example["story"])
    question = example["question"]

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"{FOOTER_OQ_COMPLETE}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_OQ_COMPLETE},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_ToMChallenges_EbE(event, memory, question=None, ebe_type="standard"):
    if question:
        system_message = SYSTEM_MESSAGE_OQ_EbE_FINAL

        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"{FOOTER_OQ_EbE_FINAL}"
        )

    else:
        if ebe_type == "obj_location":
            system_message = SYSTEM_MESSAGE_EbE_OBJ_LOCATIONS
        elif ebe_type == "location":
            system_message = SYSTEM_MESSAGE_EbE_LOCATIONS
        elif ebe_type == "belief":
            system_message = SYSTEM_MESSAGE_EbE_BELIEFS
        elif ebe_type == "task_specific":
            system_message = SYSTEM_MESSAGE_EbE_TASK_SPECIFIC
        else:
            system_message = SYSTEM_MESSAGE_EbE

        user_content = (
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_EbE}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message
    





############
#   ToMi   #
############

def build_prompt_ToMi(example):
    story = " ".join(example["story"])
    question = example["question"]

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"{FOOTER_OQ}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_OQ},
        {"role": "user", "content": user_content}
    ]
    
    return message



def build_prompt_ToMi_CoT(example):
    story = " ".join(example["story"])
    question = example["question"]

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"{FOOTER_OQ_CoT}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_OQ_CoT},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_ToMi_Complete(example):
    story = " ".join(example["story"])
    question = example["question"]

    user_content = (
        f"Story: {story}\n"
        f"Question: {question}\n"
        f"{FOOTER_OQ_COMPLETE}"
    )

    message = [
        {"role": "system", "content": SYSTEM_MESSAGE_OQ_COMPLETE},
        {"role": "user", "content": user_content}
    ]

    return message



def build_prompt_ToMi_EbE(event, memory, question=None, ebe_type="standard"):
    if question:
        system_message = SYSTEM_MESSAGE_OQ_EbE_FINAL

        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"{FOOTER_OQ_EbE_FINAL}"
        )

    else:
        if ebe_type == "obj_location":
            system_message = SYSTEM_MESSAGE_EbE_OBJ_LOCATIONS
        elif ebe_type == "location":
            system_message = SYSTEM_MESSAGE_EbE_LOCATIONS
        elif ebe_type == "belief":
            system_message = SYSTEM_MESSAGE_EbE_BELIEFS
        elif ebe_type == "task_specific":
            system_message = SYSTEM_MESSAGE_EbE_TASK_SPECIFIC
        else:
            system_message = SYSTEM_MESSAGE_EbE

        user_content = (
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_EbE}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message







#############
#   Judge   #
#############

def build_prompt_Judge(log, story_so_far):
    new_event = log.get('event_text')
    past_memory = log.get('past_memory')
    updated_memory = log.get('new_memory')

    display_story = story_so_far if story_so_far.strip() else "[This is the beginning of the story, no prior events]"

    display_past_memory = past_memory if (past_memory and past_memory.strip()) else "[No prior memory]"

    '''system_message = (
        "You are an expert judge specialized in Theory of Mind.\n"
        "Your task is to evaluate the corectness of the 'New_memory' based on the ground truth 'Story', the 'New_event', and the character's 'Past_memory'.\n"
        "RULES:\n"
        "1. PHYSICAL PRESENCE: Verify that all the locations of the objects and character are correctly stated in the 'New_memory' according to the 'Story' and 'New_evenet'.\n"
        "2. FIRST EVENT: If 'Story' and 'Past_memory' are marked as empty or [beginning of story], judge the New_memory based solely on the 'New_event'.\n"
        "3. BELIEF STASIS: A character's belief must only update if they observed the event. If a person is in another room or looking away, their belief must remain unchanged (False Belief), even if the object has moved in reality.\n"
        "4. REASONING: Provide a concise, step-by-step logical deduction of the events and beliefs first.\n"
        "5. FORMAT: Based on your reasoning, output exactly CORRECT or INCORRECT inside \\answer{{}}.\n"
        "EXAMPLE: [Step-by-step reasoning...] Therefore, the memory update is \\answer{{CORRECT}}.\n"
    )'''


    system_message = (
        "You are an expert judge specialized in Theory of Mind.\n"
        "Your task is to evaluate the corectness of the 'New_memory' based on the ground truth 'Story', the 'New_event', and the character's 'Past_memory'.\n"
        "RULES:\n"
        "1. FIRST EVENT: If 'Story' and 'Past_memory' are empty or [beginning of story], judge the corectness of 'New_memory' based solely on the 'New_event'.\n"
        "2. PHYSICAL PRESENCE: All the current locations of any object and character must be correctly stated in the 'New_memory' according to the 'Story' and 'New_event'.\n"
        "3. DIRECT OBSERVATION: A character's memory must update if they are in the same location where an event occurs. They observed the event.\n"
        "4. BELIEF STASIS: If an event occurs in a location where the character is NOT present, their memory must not update. They maintain their last known belief (False Belief).\n"
        "5. DISPLACEMENT: If a character sees an object move from A to B, the memory must reflect the NEW location and remove the OLD location. If they don't see the move, they must still believe it is at A.\n"
        "6. EXIT STATES: If a character leaves a room, they still remember who/what was in that room when they left, but they cannot know if those things move after they exit.\n"
        "7. NO INFERENCE: Characters do not 'guess' or 'assume' changes happened elsewhere. They are strictly limited to what they have seen or were told.\n\n"
        "8. STATE DELETION: When a character observes a change (e.g., someone leaves), the old state must be removed. The 'New_memory' should not contain conflicting locations for the same person/object.\n"
        "EXAMPLE: [Step-by-step reasoning...] Therefore, the memory update is \\answer{{CORRECT}}.\n"
    )

    user_content = (
        f"Story: {display_story}\n"
        f"New_event: {new_event}\n"
        f"Past_memory: {display_past_memory}\n"
        f"New_memory: {updated_memory}\n"
        "Reason step by step and output your decision, CORRECT or INCORRECT, inside \\answer{{}}."
    )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_Judge_location(log):
    updated_memory = log.get('new_memory')

    system_message = (
        "You are a rigorous judge.\n"
        "Your task is to evaluate the 'Memory' and analyze if the location of each object and person is present in it.\n"
        "RULES:\n"
        "1. SCOPE: Identify every person and object mentioned in the 'Memory', and check if the text explicitly states its current location."
        "2. FORMAT: Based on your analysis, output CORRECT if all the entities (objects and people) have a current location, or INCORRECT, otherwise. Output CORRECT or INCORRECT inside \\answer{{}}.\n"
        "EXAMPLE: [Analysis...] Therefore, the 'Memory' is \\answer{{CORRECT}}.\n"
    )

    user_content = (
        f"Memory: {updated_memory}\n"
        "Analyze the Memory and output your decision, CORRECT or INCORRECT, inside \\answer{{}}."
    )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_Judge_belief(log):
    updated_memory = log.get('new_memory')

    system_message = (
        "You are a rigorous Theory of Mind judge.\n"
        "Your task is to evaluate the 'Memory' and analyze if there is a belief associated to each people present in it.\n"
        "RULES:\n"
        "1. SCOPE: Identify every person mentioned in the 'Memory', and check if the text explicitly states its belief."
        "2. FORMAT: Based on your analysis, output CORRECT if all the people have a belief, or INCORRECT, otherwise. Output CORRECT or INCORRECT inside \\answer{{}}.\n"
        "EXAMPLE: [Analysis...] Therefore, the 'Memory' is \\answer{{CORRECT}}.\n"
    )

    user_content = (
        f"Memory: {updated_memory}\n"
        "Analyze the Memory and output your decision, CORRECT or INCORRECT, inside \\answer{{}}."
    )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message







####################
#   with History   #
####################


SYSTEM_MESSAGE_wH = (
    "You are a helpful assistant specialized in Theory of Mind.\n"
    "Your task is to update the 'Past Memory' based on the 'New Event' and the 'Prior story', and return the new complete, updated memory.\n"
    "RULES:\n"
    "1. STATE OVERWRITE: For every entity (person or object), maintain only their current, most recent state. If a 'New Event' changes an entity's location or belief, delete the old fact and replace it with the new one.\n"
    "2. SCOPE: Track the physical locations of objects/people and the internal mental states (beliefs, knowledge, intentions) of all characters.\n"
    "3. SUMMARY: Do not transcribe the 'New Event' or 'Prior story' text in the memory. Extract and store only the essential state changes."
    "4. FORMAT: Return ONLY the updated memory inside \\memory{{}}.\n"
    "EXAMPLE: \\memory{{The ball is in the box; Anne is in the room and believes the ball is in the box; Bob is in the garden and believes the ball is in the basket.}}.\n"
)

FOOTER_wH = "Output the updated memory inside \\memory{{}}."



SYSTEM_MESSAGE_MC_wH_FINAL = (
    "You are a helpful assistant specialized in Theory of Mind."
    "Your task is to select the correct option based on the provided memory.\n"
    "RULES:\n"
    "1. FORMAT: Output ONLY the letter of the chosen option inside \\answer{{}}.\n"
    "EXAMPLE: If the correct option is 'A. The kitchen', your output must be \\answer{{A}}.\n"
)

FOOTER_MC_wH_FINAL = "You must output ONLY the chosen option letter inside \\answer{{}}."


SYSTEM_MESSAGE_OQ_wH_FINAL = (
    "You are a helpful assistant specialized in Theory of Mind."
    "Your task is to provide the correct answer based on the provided memory.\n"
    "RULES:\n"
    "1. FORMAT: Output ONLY the answer, ONE WORD, inside \\answer{{}}.\n"
    "EXAMPLE: If the correct answer is 'the fridge', your output must be \\answer{{fridge}}.\n"
)

FOOTER_OQ_wH_FINAL = "You must output ONLY the answer, ONE WORD, inside \\answer{{}}."




def build_prompt_BigToM_wH(event, memory, history, question=None, shuffled_answers=None):
    if question:
        system_message = SYSTEM_MESSAGE_MC_wH_FINAL

        options = f"\nOptions:\nA. {shuffled_answers[0]}\nB. {shuffled_answers[1]}"
        question = f"{question}{options}"

        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"{FOOTER_MC_wH_FINAL}"
        )

    else:
        system_message = SYSTEM_MESSAGE_wH

        user_content = (
            f"Prior story:\n{history}\n"
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_wH}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_HiToM_wH(event, memory, history, question=None, options=None):
    if question:
        system_message = SYSTEM_MESSAGE_MC_wH_FINAL

        option_lines = "\n".join(options)
        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"Options: {option_lines}\n"
            f"{FOOTER_MC_wH_FINAL}"
        )

    else:
        system_message = SYSTEM_MESSAGE_wH


        user_content = (
            f"Prior story:\n{history}\n"
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_wH}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message


def build_prompt_ToMChallenges_wH(event, memory, history, question=None):
    if question:
        system_message = SYSTEM_MESSAGE_OQ_wH_FINAL

        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"{FOOTER_OQ_wH_FINAL}"
        )

    else:
        system_message = SYSTEM_MESSAGE_wH

        user_content = (
            f"Prior story:\n{history}\n"
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_wH}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message

def build_prompt_ToMi_wH(event, memory, history, question=None):
    if question:
        system_message = SYSTEM_MESSAGE_OQ_wH_FINAL

        user_content = (
            f"Memory: {memory}\n"
            f"Question: {question}\n"
            f"{FOOTER_OQ_wH_FINAL}"
        )

    else:
        system_message = SYSTEM_MESSAGE_wH

        user_content = (
            f"Prior story:\n{history}\n"
            f"New Event: {event}\n"
            f"Past Memory: \\memory{memory}\n"
            f"{FOOTER_wH}"
        )

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    return message
    