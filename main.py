from environs import Env
import matplotlib
from playsound import playsound

from aituber.agents import Agent, Alice, Berry


def main(_: Env):
    agents: list[Agent] = [
        Alice(),
        Berry(),
    ]
    last_message: str = "適当な話題を出して"

    # for i, user_message in enumerate(user_messages):
    for i in range(1):
        a1 = agents[i % 2]
        a2 = agents[(i+1) % 2]
        last_message = a1.chat(last_message, a2.emotion)
        mp3_path = a1.voice_read(last_message)
        playsound(mp3_path)
        print(f"{a1.name}: {last_message}")
        print(f"- emotion: {a1.emotion}")


if __name__ == "__main__":
    env = Env()
    env.read_env()  # read .env
    matplotlib.use("TkAgg")
    main(env)