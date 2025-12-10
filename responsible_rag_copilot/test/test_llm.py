from llm_utils import chat_once

def main():
    system_prompt = "You are a helpful assistant."
    user_message = "Say hello in one short sentence."
    reply = chat_once(system_prompt, user_message)
    print("Model reply:")
    print(reply)

if __name__ == "__main__":
    main()