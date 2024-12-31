from __future__ import annotations
import io
import json
from strenum import StrEnum
from typing import Optional, TypedDict

import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from PIL import Image


class Model(StrEnum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

class Emotion(StrEnum):
    HAPPY = "幸せ"
    JOY = "喜び"
    SAD = "悲しい"
    ANGRY = "怒り"
    FEAR = "恐怖"
    SURPRISE = "驚き"

    @classmethod
    def make_dict(cls) -> EmotionDict:
        return {e: 0 for e in cls}

EmotionDict = dict[Emotion, int]


class ChatEngineState(TypedDict):
    sentence: str
    emotion: EmotionDict = Emotion.make_dict()


class ChatEngine:
    def __init__(
        self,
        model: Model = Model.GPT_4O_MINI,
        system_prompt: str = "",
        temperature: float = 0.7,
        llm_invoke_retry_count: int = 3
    ):
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.llm_invoke_retry_count = llm_invoke_retry_count
        self.init_state: Optional[ChatEngineState] = None

        self.graph = StateGraph(ChatEngineState)
        self.graph.add_node("invoke_llm", self.invoke_llm)
        self.graph.set_entry_point("invoke_llm")
        self.graph.add_edge("invoke_llm", END)
        self.csg = self.graph.compile()

        self.message_history: list[BaseMessage] = [
            SystemMessage(system_prompt),
        ]

    def send(
        self,
        sentence: str,
        emotion: EmotionDict,
    ) -> ChatEngineState:
        state = ChatEngineState(sentence=sentence, emotion=emotion)
        new_state = self.csg.invoke(state)
        return new_state

    def invoke_llm(self, state: ChatEngineState) -> ChatEngineState:
        self.message_history.append(HumanMessage(state["sentence"]))

        content_json: Optional[dict[str, dict[str, int]]] = None
        for i in range(self.llm_invoke_retry_count):
            llm_responce = self.llm.invoke(self.message_history)
            llm_sentence = llm_responce.content.strip()
            try:
                content_json = json.loads(llm_sentence)
                break
            except json.JSONDecodeError:
                self.message_history.pop()
                new_sentence = state["sentence"] + \
"\n指定されたJsonフォーマットでの間違えないように細心の注意を払って返答せよ。"
                self.message_history.append(HumanMessage(new_sentence))
                print(f"Json parse error found {i+1} times. Retry LLM invole...")
                print(llm_sentence)
        if content_json is None:
            raise ValueError("Failed to parse json from LLM response "
                             f"{self.llm_invoke_retry_count} times...")

        new_state = ChatEngineState(
            sentence=content_json["sentence"],
            emotion=content_json["emotion"],
        )
        self.message_history.append(AIMessage(content_json["sentence"]))
        return new_state

    # def adjust_emotion(self, emotions: List[str]) -> str:
    #     emotion_weights = {"happy": 1, "sad": -1, "angry": -2, "neutral": 0}
    #     weighted_sum = sum(emotion_weights[e] for e in emotions[-3:])  # 直近3つの感情を考慮
    #     if weighted_sum > 1:
    #         return "happy"
    #     elif weighted_sum < -1:
    #         return "sad"
    #     else:
    #         return "neutral"

    def visualize_graph(self, csg: CompiledStateGraph):
        graph_png_binary = csg.get_graph().draw_mermaid_png()
        graph_png = Image.open(io.BytesIO(graph_png_binary))
        plt.figure(figsize=(10, 8))
        plt.imshow(graph_png)
        plt.axis('off')  # 軸を非表示にする
        plt.show()


