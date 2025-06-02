import random

def ask_question(question, answer):
    user_answer = input(f"{question} = ")
    try:
        if int(user_answer) == answer:
            print("Correct! You may proceed.\n")
            return True
        else:
            print(f"Oops! The correct answer was {answer}. Try the next one.\n")
            return False
    except ValueError:
        print("Please enter a valid number.")
        return False

def game():
    print("Welcome to Math Quest: Treasure Island!")
    print("Solve math puzzles to reach the treasure.\n")

    locations = [
        ("Beach", "What is 23 + 19?", 23 + 19),
        ("Jungle", "What is 56 - 17?", 56 - 17),
        ("Cave", "What is 7 x 6?", 7 * 6),
        ("Mountain", "What is 48 ÷ 8?", 48 // 8),
        ("Waterfall", "If you have 3 bags with 12 apples each, how many apples?", 3 * 12)
    ]

    for location, question, answer in locations:
        print(f"You arrive at the {location}!")
        while not ask_question(question, answer):
            pass  # Keep asking until correct

    print("Congratulations! You found the treasure and completed Math Quest!")

if __name__ == "__main__":
    game()